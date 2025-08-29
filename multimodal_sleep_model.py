"""
다중 모달(PPG/ECG) 수면 단계 분류 모델
- 두 모델 제공:
  1) SleepPPGNet (PPG-only)
  2) MultiModalSleepNet (PPG + ECG fusion)
- 처리 흐름(공통):
  ResConv ×8(feature 추출) → 4800을 (1200×4)로 재배치 → window별 FC(→128) → TCN ×2 → 1×1 Conv → (B, 4, 1200)
"""
import os
import sys
import datetime as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# 전역 VERBOSE/로깅 설정
# 본 training 시에는 'VERBOSE', 'LOG_TO_FILE' 모두 False로 설정
VERBOSE = False  # <-- 이 값만 True/False로 바꾸면 전체 디버깅 on/off
LOG_TO_FILE = False  # 파일(저장) 로깅 사용 여부

_LOG_FH = None
def _init_logger():
    """logs 디렉토리 생성 및 파일 핸들러 준비"""
    global _LOG_FH
    if not LOG_TO_FILE:
        return
    os.makedirs("logs", exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"multimodal_sleep_model_{ts}.log")
    _LOG_FH = open(log_path, "a", buffering=1, encoding="utf-8")
    _plain_log(f"[LOG] Logging to file: {log_path}")

def _timestamp():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _plain_log(msg: str):
    """무조건 파일/콘솔로 기록 (시작 문구 용)"""
    line = f"{_timestamp()} | {msg}"
    # 콘솔
    print(line)
    # 파일
    if _LOG_FH:
        _LOG_FH.write(line + "\n")

def log(msg: str):
    """VERBOSE가 True일 때만 파일/콘솔로 기록"""
    if not VERBOSE:
        return
    line = f"{_timestamp()} | {msg}"
    print(line)
    if _LOG_FH:
        _LOG_FH.write(line + "\n")

# 로그 초기화
_init_logger()

# 디버깅 유틸 각 network 'VERBOSE' 옵션으로 설정
def _dbg_shape(msg, x):
    """중간 텐서를 (B,C,L) 출력"""
    if not VERBOSE:
        return
    if isinstance(x, torch.Tensor):
        b = x.shape[0] if x.dim() > 0 else None
        c = x.shape[1] if x.dim() > 1 else None
        l = x.shape[2] if x.dim() > 2 else None
        log(f"{msg:<28s} | shape={tuple(x.shape)}  (B={b}, C={c}, L={l})")
    else:
        log(f"{msg:<28s} | {x}")

# ResConvBlock
# - 1D Conv 3회(각각 BN+LeakyReLU)로 feature 추출
# - 마지막에 MaxPool(2)로 시간 길이를 절반으로 줄임
# - 입력/출력 채널 다르면 → 덧셈하려면 크기가 같아야 하므로 skip 경로에 1×1 Conv로 채널 맞춤
# - skip 경로도 MaxPool(2)로 시간 길이를 동일하게 맞춘 뒤 더함
# - 입력 (B, Cin, L) → 출력 (B, Cout, L/2)
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, name=None):
        super(ResConvBlock, self).__init__()
        self.name = name or f"ResConv({in_channels}->{out_channels})"
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 채널 수가 다르면 덧셈을 위해 1×1 Conv로 채널 맞춤
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        _dbg_shape(f"[{self.name}] in", x)
        skip = x  # 나중에 더할 원본

        # 메인 경로: Conv-BN-활성화 ×3
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        _dbg_shape(f"[{self.name}] conv1", x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        _dbg_shape(f"[{self.name}] conv2", x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        _dbg_shape(f"[{self.name}] conv3", x)

        # 시간 축 절반으로 줄임
        x = self.pool(x)
        _dbg_shape(f"[{self.name}] main maxpool", x)

        # skip 경로도 크기 맞춤
        if self.residual_conv is not None:
            skip = self.residual_conv(skip)
            _dbg_shape(f"[{self.name}] skip 1x1", skip)
        skip = F.max_pool1d(skip, kernel_size=2, stride=2)
        _dbg_shape(f"[{self.name}] skip maxpool", skip)

        # 동일 크기에서 더함
        out = x + skip
        _dbg_shape(f"[{self.name}] out(+)", out)
        return out

# Chomp1d
# - dilated Conv에서 padding 때문에 출력 길이가 늘어남
# - 원래 길이에 맞추려고 뒤쪽을 잘라냄
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# TemporalBlock
# - dilation이 있는 1D Conv 두 번(각각 Chomp → LeakyReLU → Dropout)
# - Conv에는 weight_norm 적용(학습 안정화 목적)
# - 입력/출력 채널이 다르면 residual에 1×1 Conv로 맞춘 뒤 더함
# - 입/출력 크기 동일: (B, C, L) → (B, C, L)
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, name=None):
        super(TemporalBlock, self).__init__()
        self.name = name or f"TCNBlock(dil={dilation})"
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        # Conv 가중치를 작은 값(평균0, 표준편차0.01)으로 초기화 → 시작할 때 과한 값 방지
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        _dbg_shape(f"[{self.name}] in", x)
        out = self.net(x)
        _dbg_shape(f"[{self.name}] net", out)
        res = x if self.downsample is None else self.downsample(x)
        y = self.relu(out + res)
        _dbg_shape(f"[{self.name}] out(+)", y)
        return y

# TemporalConvNet (TCN)
# - TemporalBlock을 dilation 1,2,4,8,16,32 순서로 쌓음(총 6층)
# - 커널=7 기본, 입력/출력 크기는 그대로 두고 수용영역만 넓힘
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=7, dropout=0.2, name="TCN"):
        super(TemporalConvNet, self).__init__()
        self.name = name
        layers = []
        num_levels = 6
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels
            out_channels = num_channels
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout,
                                     name=f"{name}-L{i+1}(dil={dilation_size})")]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        _dbg_shape(f"[{self.name}] in", x)
        x = self.network(x)
        _dbg_shape(f"[{self.name}] out", x)
        return x

# CrossModalAttention
# - PPG feature은 Query, ECG feature은 Key/Value로 사용하여 서로 비교
# - (B, C, L) → (B, L, C)로 바꾼 뒤, FC(nn.Linear)로 Q/K/V 생성
# - 채널 C를 head 수만큼 나눠(멀티헤드) 병렬로 어텐션 계산
# - Q와 K로 가중치를 만들고, 그 가중치로 V를 섞어 새 feature을 만듦
# - 마지막에 원래 PPG feature을 더해(residual) (B, C, L)로 되돌림
class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8, name="XAttn"):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        self.name = name

        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ppg_features, ecg_features):
        _dbg_shape(f"[{self.name}] ppg_in", ppg_features)
        _dbg_shape(f"[{self.name}] ecg_in", ecg_features)

        batch_size, channels, length = ppg_features.shape

        # (B, C, L) → (B, L, C)
        ppg_features = ppg_features.transpose(1, 2)
        ecg_features = ecg_features.transpose(1, 2)

        # Q/K/V 만들기
        Q = self.query_proj(ppg_features)
        K = self.key_proj(ecg_features)
        V = self.value_proj(ecg_features)

        # 멀티헤드로 분할: (B, L, C) → (B, heads, L, head_dim)
        Q = Q.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)

        # head별 어텐션
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # 가중합하여 컨텍스트 만들기 → (B, L, C)로 합치기
        context = torch.matmul(probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, length, self.feature_dim)

        # 출력 투영 후 원래 PPG feature과 더해 보정
        output = self.out_proj(context)
        output = output + ppg_features

        # (B, L, C) → (B, C, L)
        output = output.transpose(1, 2)
        _dbg_shape(f"[{self.name}] out", output)
        return output

# MultiModalSleepNet (PPG + ECG)
# - 각 모달 인코더: ResConv ×8 → (B,256,4800)  (입력은 (B,1,1228800))
# - fusion 방식:
#   * concat   : 채널 256+256=512로 이어붙임
#   * attention: CrossModalAttention으로 PPG를 ECG로 보정(채널 256 유지)
#   * gated    : 전역 평균으로 게이트를 만들고 PPG/ECG를 가중합(채널 256 유지)
# - 4800을 (1200×4)로 나눠서, 4를 채널에 붙여 (C*4, 1200) 형태로 만듦
# - window별 FC로 (C*4 → 128) 축소 → TCN ×2 → 1×1 Conv(128→4)
# - 출력 (B, 4, 1200)  (학습 루프에서 (B,1200,4)로 바꿔 사용)
class MultiModalSleepNet(nn.Module):
    def __init__(self, fusion_strategy='attention'):
        super(MultiModalSleepNet, self).__init__()

        self.fusion_strategy = fusion_strategy

        # PPG 인코더
        self.ppg_blocks = nn.ModuleList([
            ResConvBlock(1,   16, name="PPG-Res1"),
            ResConvBlock(16,  16, name="PPG-Res2"),
            ResConvBlock(16,  32, name="PPG-Res3"),
            ResConvBlock(32,  32, name="PPG-Res4"),
            ResConvBlock(32,  64, name="PPG-Res5"),
            ResConvBlock(64,  64, name="PPG-Res6"),
            ResConvBlock(64, 128, name="PPG-Res7"),
            ResConvBlock(128,256, name="PPG-Res8"),
        ])

        # ECG 인코더
        self.ecg_blocks = nn.ModuleList([
            ResConvBlock(1,   16, name="ECG-Res1"),
            ResConvBlock(16,  16, name="ECG-Res2"),
            ResConvBlock(16,  32, name="ECG-Res3"),
            ResConvBlock(32,  32, name="ECG-Res4"),
            ResConvBlock(32,  64, name="ECG-Res5"),
            ResConvBlock(64,  64, name="ECG-Res6"),
            ResConvBlock(64, 128, name="ECG-Res7"),
            ResConvBlock(128,256, name="ECG-Res8"),
        ])

        # fusion 모듈 설정
        if fusion_strategy == 'concat':
            self.fusion_dim = 512  # 256 + 256
            self.fusion_layer = nn.Identity()
        elif fusion_strategy == 'attention':
            self.fusion_dim = 256  # 채널 유지
            self.fusion_layer = CrossModalAttention(256, name="XAttn")
        elif fusion_strategy == 'gated':
            self.fusion_dim = 256  # 채널 유지
            self.ppg_gate = nn.Sequential(
                nn.Linear(256, 256),
                nn.Sigmoid()
            )
            self.ecg_gate = nn.Sequential(
                nn.Linear(256, 256),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion_strategy: {fusion_strategy}")

        # window 임베딩 축소: (fusion_dim×4) → 128
        self.dense = nn.Linear(self.fusion_dim * 4, 128)

        # 시간 모델링: TCN ×2
        self.tcnblock1 = TemporalConvNet(128, 128, kernel_size=7, dropout=0.2, name="TCN1")
        self.tcnblock2 = TemporalConvNet(128, 128, kernel_size=7, dropout=0.2, name="TCN2")

        # 분류 헤드
        self.final_conv = nn.Conv1d(128, 4, 1)

    def _encode_stream(self, x, blocks, tag):
        log(f"INFO 1) FE({tag}): 8×ResConv (feature 추출)")
        log("────────────────────────────────────────────")
        _dbg_shape(f"[{tag}] input", x)
        for blk in blocks:
            x = blk(x)
        _dbg_shape(f"[{tag}] encoded", x)
        log("")  # 줄바꿈
        return x

    def forward(self, ppg, ecg):
        # 1) 각 모달 feature 추출
        ppg_f = self._encode_stream(ppg, self.ppg_blocks, "PPG")
        ecg_f = self._encode_stream(ecg, self.ecg_blocks, "ECG")

        # 2) fusion
        if self.fusion_strategy == 'concat':
            log("INFO 2) Fusion: concat (PPG||ECG → C=512)")
            log("────────────────────────────────────────────")
            fused = torch.cat([ppg_f, ecg_f], dim=1)  # (B, 512, 4800)
            _dbg_shape("[FUSION concat]", fused)
        elif self.fusion_strategy == 'attention':
            log("INFO 2) Fusion: cross-modal attention (C=256 유지)")
            log("────────────────────────────────────────────")
            fused = self.fusion_layer(ppg_f, ecg_f)   # (B, 256, 4800)
            _dbg_shape("[FUSION attn]", fused)
        else:  # gated
            log("INFO 2) Fusion: gated (global avg → gate, C=256 유지)")
            log("────────────────────────────────────────────")
            ppg_gate = self.ppg_gate(ppg_f.mean(dim=2)).unsqueeze(2)  # (B,256,1)
            ecg_gate = self.ecg_gate(ecg_f.mean(dim=2)).unsqueeze(2)  # (B,256,1)
            _dbg_shape("[PPG gate]", ppg_gate)
            _dbg_shape("[ECG gate]", ecg_gate)
            fused = ppg_gate * ppg_f + ecg_gate * ecg_f               # (B,256,4800)
            _dbg_shape("[FUSION gated]", fused)
        log("")

        # 3) 4800 → (1200×4)로 나눈 뒤 4를 채널에 붙임
        B, C, L = fused.shape  # (B, 256, 4800) 예상
        win_len = 1200
        n_win = L // win_len   # = 4 (윈도 개수)
        assert L % win_len == 0, f"Window len({win_len}) must divide length({L})"
        log(f"INFO 3) Windowing: L={L} → {win_len}×{n_win} (윈도 개수 n_win={n_win})")
        log(f"INFO 3.1) Channel stack: C={C} × n_win={n_win} → C'={C*n_win} (예: 256×4=1024)")
        log("────────────────────────────────────────────")
        fused = fused.view(B, C, win_len, n_win)               # (B, C, 1200, 4)
        fused = fused.permute(0, 1, 3, 2)                      # (B, C, 4, 1200)
        fused = fused.contiguous().view(B, C * n_win, win_len) # (B, C*4, 1200)
        _dbg_shape("[Window -> stack x4]", fused)
        log("")

        # 4) window별 축소: (C*4 → 128)
        log("INFO 4) Dense: window별 FC (C×n_win → 128)")
        log("──────────────────────────────────────")
        fused = fused.transpose(1, 2)  # (B, 1200, C*4)
        _dbg_shape("[Pre-Dense]", fused)
        fused = self.dense(fused)      # (B, 1200, 128)
        fused = fused.transpose(1, 2)  # (B, 128, 1200)
        _dbg_shape("[Post-Dense]", fused)
        log("")

        # 5) TCN ×2
        log("INFO 5) TCN ×2: 시계열 컨텍스트 확장 (dilated conv)")
        log("──────────────────────────────────────")
        x = self.tcnblock1(fused)
        x = self.tcnblock2(x)
        log("")

        # 6) 분류
        log("INFO 6) Classifier: 1×1 Conv → Softmax")
        log("──────────────────────────────────────")
        x = self.final_conv(x)         # (B, 4, 1200)
        _dbg_shape("[Final 1x1 conv]", x)
        x = F.softmax(x, dim=1)        # 참고: CrossEntropyLoss 쓸 땐 보통 logits로 두기도 함
        _dbg_shape("[Softmax out]", x)
        log("")
        return x

# SleepPPGNet (PPG-only)
# - PPG만 사용. 위와 동일한 파이프라인에서 ECG fusion만 제외
# - (B,1,1228800) → ResConv ×8 → (B,256,4800)
#   → 4800을 (1200×4)로 → 채널×4=1024 → FC(1024→128)
#   → TCN ×2 → 1×1 Conv → (B,4,1200)
class SleepPPGNet(nn.Module):
    def __init__(self):
        super(SleepPPGNet, self).__init__()
        # 요구사항(논문):
        # ResConv 8개, 각 블록은 1D Conv ×3 → MaxPool(2) → Residual add, kernel=3,
        # 채널 [16,16,32,32,64,64,128,256],
        # MaxPool로 시간축/2,
        # 최종 4800×256 임베딩
        # 1228800 / 2^8(8개 블록) = 1228800 / 256 = 4800 -> (B, 256, 4800) 최종 텐서 크기
        self.blocks = nn.ModuleList([
            ResConvBlock(1,   16, name="Res1"),
            ResConvBlock(16,  16, name="Res2"),
            ResConvBlock(16,  32, name="Res3"),
            ResConvBlock(32,  32, name="Res4"),
            ResConvBlock(32,  64, name="Res5"),
            ResConvBlock(64,  64, name="Res6"),
            ResConvBlock(64, 128, name="Res7"),
            ResConvBlock(128,256, name="Res8"),
        ])

        # 1024(=256×4) → 128
        self.dense = nn.Linear(1024, 128)

        self.tcnblock1 = TemporalConvNet(128, 128, kernel_size=7, dropout=0.2, name="TCN1")
        self.tcnblock2 = TemporalConvNet(128, 128, kernel_size=7, dropout=0.2, name="TCN2")

        self.final_conv = nn.Conv1d(128, 4, 1)

    def forward(self, x):
        log("INFO 1) FE: 8×ResConv (feature 추출)")
        log("────────────────────────────────────────────")
        _dbg_shape("[PPG input]", x)

        # (B,1,1228800) → (B,256,4800)
        for blk in self.blocks:
            x = blk(x)
        _dbg_shape("[After 8×ResConv]", x)
        log("")

        # 4800 → (1200×4) → 채널로 합치기
        B, C, L = x.shape
        win_len = 1200
        n_win = L // win_len
        assert L % win_len == 0, f"Window len({win_len}) must divide length({L})"
        log(f"INFO 2) Windowing: L={L} → {win_len}×{n_win} (윈도 개수 n_win={n_win})")
        log(f"INFO 2.1) Channel stack: C={C} × n_win={n_win} → C'={C*n_win} (예: 256×4=1024)")
        log("────────────────────────────────────────────")
        x = x.view(B, C, win_len, n_win)                      # (B,256,1200,4)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C*n_win, win_len)  # (B,1024,1200)
        _dbg_shape("[Window -> stack x4]", x)
        log("")

        # (1024 → 128)
        log("INFO 3) Dense: window별 FC (C×n_win → 128)")
        log("──────────────────────────────────────")
        x = x.transpose(1, 2)  # (B,1200,1024)
        _dbg_shape("[Pre-Dense]", x)
        x = self.dense(x)      # (B,1200,128)
        x = x.transpose(1, 2)  # (B,128,1200)
        _dbg_shape("[Post-Dense]", x)
        log("")

        # TCN ×2
        log("INFO 4) TCN ×2: 시계열 컨텍스트 확장 (dilated conv)")
        log("──────────────────────────────────────")
        x = self.tcnblock1(x)
        x = self.tcnblock2(x)
        log("")

        # 분류
        log("INFO 5) Classifier: 1×1 Conv → Softmax")
        log("──────────────────────────────────────")
        x = self.final_conv(x)  # (B,4,1200)
        _dbg_shape("[Final 1x1 conv]", x)
        x = F.softmax(x, dim=1)
        _dbg_shape("[Softmax out]", x)
        log("")
        return x

# 간단 동작 확인(형상만)
def test_models():
    batch_size = 2
    ppg = torch.randn(batch_size, 1, 1228800)
    ecg = torch.randn(batch_size, 1, 1228800)

    log("Testing PPG-only model...")
    ppg_model = SleepPPGNet()
    ppg_out = ppg_model(ppg)
    print(f"PPG-only output shape: {ppg_out.shape}")  # (2, 4, 1200)

    # 필요시 멀티모달 테스트를 직접 추가해서 사용
    # log("\nTesting MultiModal model...")
    # mm_model = MultiModalSleepNet(fusion_strategy='attention')
    # mm_out = mm_model(ppg, ecg)
    # print(f"MultiModal output shape: {mm_out.shape}")  # (2, 4, 1200)

    # 파라미터 수(참고)
    ppg_params = sum(p.numel() for p in ppg_model.parameters())
    print(f"\nPPG-only parameters: {ppg_params:,}")
    # mm_params = sum(p.numel() for p in mm_model.parameters())
    # print(f"MultiModal parameters: {mm_params:,}")

if __name__ == "__main__":
    try:
        test_models()
    finally:
        if _LOG_FH:
            _LOG_FH.close()