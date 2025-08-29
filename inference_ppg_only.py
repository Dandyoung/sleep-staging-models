#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPG-only inference script (with cache reuse)
- 최소 inference 블록(only_inference / edf_file / random_seed) 지원
- H5(라벨 有) 분석 모드: 테스트셋 교집합에서 n_subjects명 샘플링
- EDF(라벨 無) 모드: 단일 EDF 파일 전처리→추론 (캐시 있으면 반드시 재사용)

출력 디렉토리:
  <inference.save_dir or /workspace/NSRR/sleep-staging-models/outputs/inference>/<RUN_ID>/
    ├── result.json
    ├── csv/<ID>/<ID>.csv
    └── graph/<ID>/<ID>.png                        # 예측만 (항상)
        graph/<ID>/<ID>_with_label.png             # 분석(H5) 모드에서만: 검은 라인 + 빨간 X(불일치)

모델 체크포인트:
  --checkpoint 또는 inference.best_model (필수)
"""

import os, json, argparse, traceback, warnings
from pathlib import Path
import datetime as dt
warnings.filterwarnings("ignore")

import numpy as np
import h5py
import yaml
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# EDF
import pyedflib
from scipy import signal
from scipy.signal import cheby2

# 모델 (PPG-only)
from multimodal_sleep_model import SleepPPGNet

# 고정 테스트셋
from multimodal_dataset_aligned import SLEEPPPG_TEST_SUBJECTS

# ─────────────────────────────────────────────────────────────
# 상수/유틸
# ─────────────────────────────────────────────────────────────
STAGE_NAMES = {-1: "UNK", 0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
STAGES_ORDER = ["Wake", "Light", "Deep", "REM"]
NUM_CLASSES = 4
WIN_PER_SUBJ = 1200
SAMPLES_PER_WIN = 1024
TARGET_LEN = WIN_PER_SUBJ * SAMPLES_PER_WIN  # 1,228,800
TARGET_FS = 34.133333333
WINDOW_SEC = 30

# 캐시 루트 (기본)
DEFAULT_CACHE_ROOT = "./mesa-inference"
CONT_NAME = "ppg_continuous.npy"
WIN_NAME  = "ppg_windows.npy"
META_NAME = "meta.json"

def now_str():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def subject_id_from_edf(edf_path: str) -> str:
    base = os.path.basename(edf_path)
    try:
        return base.split('-')[2].split('.')[0]  # mesa-sleep-0010.edf -> 0010
    except Exception:
        return base.replace(".edf", "")

def make_time_hours(num_epochs: int) -> np.ndarray:
    return np.arange(num_epochs) / 120.0  # 30초 = 1/120 시간

def kappa_acc(y_true, y_pred):
    from sklearn.metrics import cohen_kappa_score, accuracy_score
    return float(cohen_kappa_score(y_true, y_pred)), float(accuracy_score(y_true, y_pred))

# ─────────────────────────────────────────────────────────────
# EDF 전처리 (학습과 동일)
# ─────────────────────────────────────────────────────────────
def preprocess_ppg_from_edf(
    edf_file: str,
    target_fs: float = TARGET_FS,
    window_duration: int = WINDOW_SEC,
    samples_per_window: int = SAMPLES_PER_WIN,
    target_windows: int = WIN_PER_SUBJ,
):
    """
    반환:
      ppg_continuous (1228800,), ppg_windows (1200,1024), meta(dict)
    """
    f = pyedflib.EdfReader(edf_file)

    # 채널 라벨 탐색
    labels = [l.lower() for l in f.getSignalLabels()]
    ppg_idx = None
    for idx, lab in enumerate(labels):
        if ("pleth" in lab) or ("ppg" in lab):
            ppg_idx = idx
            break
    if ppg_idx is None:
        f.close()
        raise RuntimeError(f"No PPG channel found in EDF: {edf_file}")

    sig = f.readSignal(ppg_idx)
    orig_fs = f.getSampleFrequency(ppg_idx)
    ch_label = f.getSignalLabels()[ppg_idx]
    f.close()

    # Lowpass @ 8 Hz, zero-phase
    nyq = 0.5 * orig_fs
    cutoff = 8.0 / nyq
    sos = cheby2(N=8, rs=40, Wn=cutoff, btype='lowpass', output='sos')
    filtered = signal.sosfiltfilt(sos, sig)

    # Resample via interpolation
    duration = len(filtered) / orig_fs
    n_samples = int(duration * target_fs)
    old_idx = np.linspace(0, len(filtered) - 1, len(filtered))
    new_idx = np.linspace(0, len(filtered) - 1, n_samples)
    resampled = np.interp(new_idx, old_idx, filtered)

    # Clip ±3σ → z-score
    mu, sd = np.mean(resampled), np.std(resampled)
    clipped = np.clip(resampled, mu - 3 * sd, mu + 3 * sd)
    z = (clipped - np.mean(clipped)) / (np.std(clipped) + 1e-8)

    # pad/trunc to TARGET_LEN
    if len(z) >= TARGET_LEN:
        z = z[:TARGET_LEN]
    else:
        z = np.concatenate([z, np.zeros(TARGET_LEN - len(z), dtype=z.dtype)])

    windows = z.reshape(target_windows, samples_per_window)

    meta = {
        "edf_file": os.path.abspath(edf_file),
        "ppg_channel_label": ch_label,
        "original_fs": float(orig_fs),
        "target_fs": float(target_fs),
        "window_duration_sec": int(window_duration),
        "samples_per_window": int(samples_per_window),
        "target_windows": int(target_windows),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    return z, windows, meta

def load_cache_or_preprocess(edf_path: str, cache_root: str = DEFAULT_CACHE_ROOT):
    """캐시가 있으면 항상 재사용. 없으면 전처리 후 캐시에 저장."""
    sid = subject_id_from_edf(edf_path)
    cache_dir = os.path.join(cache_root, sid)
    cont_path = os.path.join(cache_dir, CONT_NAME)
    win_path  = os.path.join(cache_dir, WIN_NAME)
    meta_path = os.path.join(cache_dir, META_NAME)

    # 캐시 재사용 시도
    if os.path.exists(cont_path) and os.path.exists(win_path):
        try:
            ppg_cont = np.load(cont_path)
            ppg_win  = np.load(win_path)
            if ppg_cont.shape[0] == TARGET_LEN and ppg_win.shape == (WIN_PER_SUBJ, SAMPLES_PER_WIN):
                meta = {}
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                return sid, ppg_cont, ppg_win, meta, cache_dir, True
        except Exception:
            # 캐시가 손상됐으면 무시하고 새로 만든다.
            pass

    # 캐시 없거나 손상 → 전처리
    ppg_cont, ppg_win, meta = preprocess_ppg_from_edf(edf_path)
    ensure_dir(cache_dir)
    np.save(cont_path, ppg_cont)
    np.save(win_path,  ppg_win)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return sid, ppg_cont, ppg_win, meta, cache_dir, False

# ─────────────────────────────────────────────────────────────
# 모델 로딩 & 추론
# ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = SleepPPGNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model

def run_model_ppg_only(model, device, ppg_continuous: np.ndarray):
    """
    입력: ppg_continuous (1228800,)
    반환: probs (1200,4), preds (1200,)
    """
    x = torch.from_numpy(ppg_continuous.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,L)
    x = x.to(device)
    with torch.no_grad():
        out = model(x) # (1,4,1200)

        # # 이중 softmax 확인
        # soft = out.softmax(dim=1)
        # print(f"[smx] rawΣ={out.sum(1).mean():.3f}  raw[min,max]=[{out.min():.3f},{out.max():.3f}]  smxΣ={soft.sum(1).mean():.3f}")

        if out.dim() != 3 or out.size(1) != NUM_CLASSES:
            raise RuntimeError(f"Unexpected model output shape: {tuple(out.shape)}")
        probs = F.softmax(out, dim=1)               # softmax
        probs = probs.squeeze(0).permute(1, 0).cpu().numpy()  # (1200,4)
        preds = np.argmax(probs, axis=1)            # (1200,)
        return probs, preds

# ─────────────────────────────────────────────────────────────
# 저장/시각화
# ─────────────────────────────────────────────────────────────
def save_csv(csv_path: str, probs: np.ndarray, preds: np.ndarray, labels: np.ndarray | None):
    ensure_dir(os.path.dirname(csv_path))
    epochs = np.arange(len(preds))
    time_hours = make_time_hours(len(preds))
    true_idx = np.full_like(preds, -1, dtype=int) if labels is None else labels.astype(int)

    pred_names = [STAGE_NAMES[int(i)] for i in preds]
    true_names = [STAGE_NAMES[int(i)] for i in true_idx]
    mismatch = ((preds != true_idx) & (true_idx >= 0)).astype(int)  # EDF 모드면 전부 0

    arr = np.column_stack([
        epochs,
        time_hours,
        preds,
        np.array(pred_names, dtype=object),
        true_idx,
        np.array(true_names, dtype=object),
        mismatch,
        probs[:,0], probs[:,1], probs[:,2], probs[:,3],
    ])
    header = "epoch,time_hours,pred_idx,pred_name,true_idx,true_name,mismatch,p_wake,p_light,p_deep,p_rem"
    np.savetxt(csv_path, arr, delimiter=",", fmt="%s", header=header, comments="", encoding="utf-8")

def plot_hypnogram_png(png_path: str, preds: np.ndarray, labels: np.ndarray | None, edf_mode: bool):
    """
    - edf_mode=True: 검은 라인(예측)만
    - edf_mode=False: 검은 라인(예측) + 불일치 지점만 빨간 'x', 라벨 라인은 그리지 않음
    """
    ensure_dir(os.path.dirname(png_path))
    time_hours = make_time_hours(len(preds))

    plt.figure(figsize=(12, 3.2))
    # 예측 라인(검은색)
    plt.step(time_hours, preds, where="post", linewidth=1.8, color="black", label="Prediction")

    title = "Hypnogram (Prediction)"
    if not edf_mode and labels is not None:
        valid = labels >= 0
        if valid.any():
            from sklearn.metrics import cohen_kappa_score, accuracy_score
            y_true = labels[valid]
            y_pred = preds[valid]
            kappa = cohen_kappa_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            # 불일치 지점에만 빨간 'x'
            mism_idx = np.where((preds != labels) & (labels >= 0))[0]
            if len(mism_idx) > 0:
                plt.scatter(time_hours[mism_idx], preds[mism_idx], marker='x', s=18, color="red", label="Correction")
            title = f"SleepPPG-Net Scored Hypnogram (k={kappa:.2f} , Ac={int(round(acc*100))}%)"

    plt.yticks([0,1,2,3], STAGES_ORDER)
    plt.xlabel("Time (Hours)")
    plt.ylabel("")  # 깔끔하게
    plt.title(title)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.10),  # y를 1보다 크게 → 살짝 위로
        borderaxespad=0.3,
    )
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────
# H5 접근 (라벨 有)
# ─────────────────────────────────────────────────────────────
def get_valid_subjects(index_file: str, windows_per_subject: int = WIN_PER_SUBJ):
    subs = []
    with h5py.File(index_file, "r") as f:
        for subj in list(f["subjects"].keys()):
            nwin = int(f[f"subjects/{subj}"].attrs.get("n_windows", 0))
            if nwin == windows_per_subject:
                subs.append(subj)
    return subs

def subject_start_index(index_file: str, subj: str) -> int:
    with h5py.File(index_file, "r") as f:
        idx = f[f"subjects/{subj}/window_indices"][:]
        return int(idx[0])

def load_ppg_labels_for_subject(
    ppg_file: str, subj: str, start_idx: int,
    windows_per_subject=WIN_PER_SUBJ, samples_per_window=SAMPLES_PER_WIN
):
    with h5py.File(ppg_file, "r") as f:
        ppg_windows = f["ppg"][start_idx:start_idx+windows_per_subject]
        labels = f["labels"][start_idx:start_idx+windows_per_subject]
    ppg_cont = ppg_windows.reshape(-1)
    assert ppg_cont.shape[0] == windows_per_subject * samples_per_window
    return ppg_cont, labels

# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/config_cloud.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (.pth). Overrides inference.best_model")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    inf_cfg  = cfg.get("inference", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    # 체크포인트
    model_ckpt = args.checkpoint or inf_cfg.get("best_model")
    if not model_ckpt:
        raise ValueError("모델 체크포인트를 지정해주세요: --checkpoint 또는 inference.best_model")

    # 모드
    only_inference = bool(inf_cfg.get("only_inference", False))
    edf_file = inf_cfg.get("edf_file", None)

    # 공통 설정
    n_subjects    = int(inf_cfg.get("n_subjects", 1))  # 분석(H5) 모드에서 사용. 기본 1
    save_dir_root = inf_cfg.get("save_dir", "/workspace/NSRR/sleep-staging-models/outputs/inference")
    random_seed   = inf_cfg.get("random_seed", None)
    cache_root    = inf_cfg.get("cache_root", DEFAULT_CACHE_ROOT)

    save_csv_flag   = bool(inf_cfg.get("save_csv", True))
    save_graph_flag = bool(inf_cfg.get("save_graph", True))
    save_json_flag  = bool(inf_cfg.get("save_json", True))

    set_seed(random_seed if isinstance(random_seed, int) else None)

    # 장치/모델
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_ckpt, device)

    # 출력 폴더
    run_id = now_str()
    out_dir = os.path.join(save_dir_root, run_id)
    ensure_dir(out_dir); ensure_dir(os.path.join(out_dir, "csv")); ensure_dir(os.path.join(out_dir, "graph"))

    summary = {
        "run_id": run_id,
        "mode": "EDF" if only_inference else "H5",
        "random_seed": random_seed if isinstance(random_seed, int) else None,
        "model_checkpoint": os.path.abspath(model_ckpt),
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "items": [],
        "metrics": None,
        "errors": [],
    }

    try:
        if only_inference:
            # ───────── RAW EDF 모드 ─────────
            if not edf_file:
                raise ValueError("inference.only_inference=true 인 경우 inference.edf_file 이 필요합니다.")
            edf_path = os.path.abspath(edf_file)

            sid, ppg_cont, ppg_win, meta, cache_dir, cache_used = load_cache_or_preprocess(
                edf_path, cache_root=cache_root
            )
            probs, preds = run_model_ppg_only(model, device, ppg_cont)

            # 저장 (ID별 폴더)
            csv_path = os.path.join(out_dir, "csv", sid, f"{sid}.csv")
            png_pred = os.path.join(out_dir, "graph", sid, f"{sid}.png")
            if save_csv_flag:
                save_csv(csv_path, probs, preds, labels=None)
            if save_graph_flag:
                plot_hypnogram_png(png_pred, preds, labels=None, edf_mode=True)

            summary["items"].append({
                "subject_id": sid,
                "mode": "EDF",
                "csv": csv_path if save_csv_flag else None,
                "graph_pred": png_pred if save_graph_flag else None,
                "graph_pred_vs_label": None,
                "edf_file": edf_path,
                "cache_root": cache_root,
                "cache_dir": cache_dir,
                "cache_used": cache_used,
            })
            summary["metrics"] = None

        else:
            # ───────── 분석(H5) 모드 ─────────
            if n_subjects <= 0:
                raise ValueError(
                    f"inference.n_subjects={n_subjects} (<=0). "
                    "피실험자 수는 1명 이상이어야 합니다. configs/config_cloud.yaml 에서 설정하세요."
                )

            ppg_file   = data_cfg.get("ppg_file", "./mesa-x/mesa_ppg_with_labels.h5")
            index_file = data_cfg.get("index_file", "./mesa-x/mesa_subject_index.h5")

            # 1) 유효 subject(윈도 1200)
            all_valid = get_valid_subjects(index_file, WIN_PER_SUBJ)

            # 2) 고정 테스트셋과 교집합
            test_pool = [s for s in all_valid if s in SLEEPPPG_TEST_SUBJECTS]
            if len(test_pool) == 0:
                raise ValueError("교집합이 없습니다: 유효 subject(윈도 1200) ∩ SLEEPPPG_TEST_SUBJECTS = 0")

            # 3) 시드 셔플 & 샘플링
            rng = np.random.default_rng(random_seed if isinstance(random_seed, int) else None)
            rng.shuffle(test_pool)
            if n_subjects > len(test_pool):
                print(f"[WARN] 요청 n_subjects={n_subjects}, 사용 가능={len(test_pool)} → 가능한 만큼만 진행")
                n_subjects = len(test_pool)
            subjects = test_pool[:n_subjects]

            accs, kappas = [], []
            for sid in subjects:
                try:
                    start_idx = subject_start_index(index_file, sid)
                    ppg_cont, labels = load_ppg_labels_for_subject(
                        ppg_file, sid, start_idx, WIN_PER_SUBJ, SAMPLES_PER_WIN
                    )
                    probs, preds = run_model_ppg_only(model, device, ppg_cont)

                    # 저장 (ID별 폴더)
                    csv_path = os.path.join(out_dir, "csv",   sid, f"{sid}.csv")
                    png_pred = os.path.join(out_dir, "graph", sid, f"{sid}.png")
                    png_with = os.path.join(out_dir, "graph", sid, f"{sid}_with_label.png")

                    if save_csv_flag:
                        save_csv(csv_path, probs, preds, labels)
                    if save_graph_flag:
                        # pred-only
                        plot_hypnogram_png(png_pred, preds, labels=None, edf_mode=True)
                        # pred vs true(검은 라인 + 빨간 X)
                        plot_hypnogram_png(png_with, preds, labels, edf_mode=False)

                    # 성능
                    valid = labels >= 0
                    if valid.any():
                        k, a = kappa_acc(labels[valid], preds[valid])
                        kappas.append(k); accs.append(a)
                        perf = {"acc": a, "kappa": k}
                    else:
                        perf = {"acc": None, "kappa": None}

                    summary["items"].append({
                        "subject_id": sid,
                        "mode": "H5",
                        "csv": csv_path if save_csv_flag else None,
                        "graph_pred": png_pred if save_graph_flag else None,
                        "graph_pred_vs_label": png_with if save_graph_flag else None,
                        "ppg_file": os.path.abspath(ppg_file),
                        "index_file": os.path.abspath(index_file),
                        "performance": perf,
                    })
                except Exception as e:
                    summary["errors"].append({"subject_id": sid, "error": str(e)})
                    continue

            if len(accs) > 0:
                summary["metrics"] = {
                    "subjects": len(subjects),
                    "acc_mean": float(np.mean(accs)),
                    "acc_std": float(np.std(accs)),
                    "kappa_mean": float(np.mean(kappas)),
                    "kappa_std": float(np.std(kappas)),
                }
            else:
                summary["metrics"] = None

    except Exception as e:
        summary["errors"].append({"fatal": str(e), "trace": traceback.format_exc()})

    finally:
        summary["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        if save_json_flag:
            with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[Inference] Done. Results at: {out_dir}")

if __name__ == "__main__":
    main()
