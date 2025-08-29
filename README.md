# Sleep Staging Models - PPG 기반 수면 단계 분류 모델(SleepPPG-Net)

> **참고**: 이 프로젝트는 [DavyWJW/sleep-staging-models](https://github.com/DavyWJW/sleep-staging-models/tree/main?tab=readme-ov-file) GitHub 레포지토리를 기반으로 합니다.

## 빠른 시작

### 1. 환경 설정

> **⚠️ Python 버전**: Python 3.10.12 사용

```bash
# 저장소 클론
git clone <repository-url>
cd sleep-staging-models

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

**Argument 설명:**
- `--index-url`: PyTorch 패키지를 다운로드할 인덱스 URL
- `cu121`: CUDA 12.1 버전을 의미

> **⚠️ 중요**: H100/A100 GPU 사용 시 CUDA 12.1 버전 PyTorch 설치 필요
> - 기존 `DavyWJW/sleep-staging-models`에서 제공하는 requirements.txt의 PyTorch는 CUDA 11.x 계열로 sm_90 커널 미포함
> - H100에서 `no kernel image is available for execution on device` 에러 방지

### 2. 데이터 준비

#### 2.1 MESA 데이터셋 다운로드

**터미널 2개를 열어서 동시에 실행:**

```bash
# 터미널 1: EDF 파일 다운로드
nsrr download mesa-commercial-use/polysomnography/edfs

# 터미널 2: 어노테이션 XML 파일 다운로드  
nsrr download mesa-commercial-use/polysomnography/annotations-events-nsrr
```

**Argument 설명:**
- `mesa-commercial-use/polysomnography/edfs`: EDF 신호 파일들을 다운로드할 경로
- `mesa-commercial-use/polysomnography/annotations-events-nsrr`: XML 어노테이션 파일들을 다운로드할 경로

> **토큰 인증**: NSRR에서 받은 토큰으로 인증 진행

#### 2.2 데이터셋 검증

```bash
# EDF와 XML 파일 1:1 매핑 검증
python verify_mesa_dataset.py \
    --xml-dir ./mesa-commercial-use/polysomnography/annotations-events-nsrr \
    --edf-dir ./mesa-commercial-use/polysomnography/edfs \
    --out ./paircheck_result.json
```

**Argument 설명:**
- `--xml-dir`: XML 어노테이션 파일들이 저장된 디렉토리 경로
- `--edf-dir`: EDF 신호 파일들이 저장된 디렉토리 경로  
- `--out`: 검증 결과를 저장할 JSON 파일 경로

**검증 결과 예시:**
실행하면 root 경로 아래에 `paircheck_result.json` 생성
```json
{
  "summary": {
    "xml_count": 1900,
    "edf_count": 1900,
    "matched": 1900,
    "only_xml": 0,
    "only_edf": 0
  }
}
```

#### 2.3 데이터 전처리

```bash
# SleepPPG-Net 모델 학습을 위한 전처리 수행
python extract_mesa_data.py
```

**Argument 설명:**
- `extract_mesa_data.py`: 별도의 argument 없이 실행 (설정은 코드 내부에서 정의)

#### 2.4 전처리 결과 검증

```bash
# 전처리된 데이터셋 및 입력 길이(window 길이) 검증
python verify_mesa_outputs.py --base ./mesa-x
```

**Argument 설명:**
- `--base`: 전처리된 MESA 데이터가 저장된 기본 디렉토리 경로

### 3. 모델 training

#### SleepPPG-Net 모델로 training 시작
```bash
python train_ppg_only.py --config configs/config_cloud.yaml --model ppg_only
```

**Argument 설명:**
- `--config`: 설정 파일 경로 (YAML 형식)
- `--model`: 사용할 모델 타입 (`ppg_only`)

### 4. training 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir ./outputs --host 0.0.0.0 --port 8890

# 백그라운드 실행
nohup tensorboard --logdir ./outputs --host 0.0.0.0 --port 8890 &
```

**Argument 설명:**
- `--logdir`: TensorBoard 로그가 저장된 디렉토리 경로
- `--host`: 접속할 수 있는 호스트 주소 (0.0.0.0은 모든 인터페이스)
- `--port`: TensorBoard가 사용할 포트 번호

### 5. 모델 inference

#### PPG 전용 모델 inference
```bash
python inference_ppg_only.py --config configs/config_cloud.yaml
```

**Argument 설명:**
- `--config`: 설정 파일 경로 (모델 체크포인트, 저장 경로 등이 설정에 포함됨)

### 6. 실행 및 관리

#### 6.1 프로세스 관리

```bash
# 실행 중인 프로세스 확인
ps -ef | grep train_ppg_only.py | grep -v grep
ps -ef | grep tensorboard | grep -v grep

# 또는 PID로 확인
pgrep -fl train_ppg_only.py
pgrep -fl tensorboard
```

#### 6.2 프로세스 종료

```bash
# PID로 종료
kill <PID>

# 또는 프로세스명으로 종료
pkill -f train_ppg_only.py
pkill -f tensorboard
```

## 프로젝트 구조

```
sleep-staging-models/
├── 📁 configs/                    # 설정 파일들
│   └── config_cloud.yaml         # 클라우드 환경 설정
├── 📁 dataset/                    # 데이터셋 관련
│   └── mesa/                     # MESA 데이터셋
├── 📁 logs/                       # 로그 파일들
├── 📁 models/                     # training된 모델 체크포인트
│   └── best_model_105.pth        # 최고 성능 모델 (42MB)
├── 📁 outputs/                    # training 결과 및 체크포인트
├── 📁 venv/                       # 가상환경
├── 📁 mesa-commercial-use/        # MESA 상용 데이터셋
│   └── polysomnography/
│       ├── edfs/                  # EDF 신호 파일들 (347GB)
│       └── annotations-events-nsrr/ # XML 어노테이션 파일들 (444MB)
├── 📁 mesa-x/                     # 전처리된 MESA 데이터 (32GB)
│   ├── logs/                      # 전처리 로그
│   ├── data_stats.npy             # 데이터 통계 (numpy)
│   ├── data_stats.txt             # 데이터 통계 (텍스트)
│   ├── mesa_ppg_with_labels.h5    # PPG 데이터 + 라벨 (15GB)
│   ├── mesa_real_ecg.h5           # 실제 ECG 데이터 (16GB)
│   └── mesa_subject_index.h5      # 피실험자 인덱스 정보 (20MB)
├── 📁 mesa-inference/             # 추론 시 EDF 파일 전처리 캐시 (피실험자별)
│   └── 0010/                     # 피실험자 ID별 폴더
│       ├── meta.json             # 전처리 메타데이터 (EDF 파일 경로, 샘플링 정보 등)
│       ├── ppg_continuous.npy    # 전처리된 PPG 연속 신호 (9.4MB)
│       └── ppg_windows.npy       # 전처리된 PPG 윈도우 데이터 (9.4MB)
├── 📁 tmp/                        # 테스트 파일들
├── 📄 multimodal_sleep_model.py   # 주요 모델 아키텍처
├── 📄 train_ppg_only.py           # PPG 전용 모델 training
├── 📄 inference_ppg_only.py       # PPG 전용 모델 inference
├── 📄 extract_mesa_data.py        # MESA 데이터 추출 및 전처리
├── 📄 multimodal_dataset_aligned.py # 데이터셋 로더
├── 📄 verify_mesa_dataset.py      # 데이터셋 검증 (EDF/XML 매핑)
├── 📄 verify_mesa_outputs.py      # 전처리 결과 검증
├── 📄 paircheck_result.json       # 데이터셋 검증 결과 (203KB)
├── 📄 requirements.txt            # 의존성 패키지
└── 📄 README.md                   # 프로젝트 설명서
```

## 주요 파일 설명

### 핵심 모델 파일
- **`multimodal_sleep_model.py`**: 주요 모델 아키텍처 (SleepPPGNet 외의 다른 모델 포함)
- **`train_ppg_only.py`**: SleepPPGNet 전용 training 스크립트
- **`inference_ppg_only.py`**: SleepPPGNet 전용 모델 inference 스크립트

### 데이터 처리
- **`multimodal_dataset_aligned.py`**: 데이터셋 로더 및 배치 생성
- **`verify_mesa_dataset.py`**: EDF/XML 파일 1:1 매핑 검증
- **`verify_mesa_outputs.py`**: 전처리된 데이터셋 및 입력 길이 검증

### 데이터 검증 및 전처리
- **`verify_mesa_dataset.py`**: 다운로드된 MESA 데이터셋 검증
- **`verify_mesa_outputs.py`**: 전처리 결과 및 window 길이 확인
- **`extract_mesa_data.py`**: SleepPPG-Net 학습용 데이터 전처리

## 데이터셋 정보

| 항목 | 크기 | 설명 |
|------|------|------|
| **전체 데이터셋** | **348GB** | MESA 데이터셋 전체 크기 |
| **XML 파일** | **444MB** | 수면 단계 어노테이션 파일들 |
| **EDF 파일** | **347GB** | 생체신호 데이터 파일들 |
| **전처리 후 데이터셋** | **32GB** | 모델 학습용으로 가공된 데이터 |

### 데이터셋 상세 정보
- **총 피실험자 수**: 1,900명
- **EDF/XML 매칭**: 1,900개 (100% 매칭)
- **전처리 후 파일** (총 5개):
  - `logs/` - 전처리 로그
  - `data_stats.npy` - 데이터 통계 (numpy)
  - `data_stats.txt` - 데이터 통계 (텍스트)
  - `mesa_ppg_with_labels.h5` - PPG 신호 + 라벨 (15GB)
  - `mesa_real_ecg.h5` - ECG 신호 (16GB)
  - `mesa_subject_index.h5` - 피실험자 인덱스 (20MB)

### 예상 소요 시간 (네트워크 속도에 따라 다를 수 있음)
- **데이터셋 다운로드**: `약 17시간`
- **데이터 전처리**: `약 1시간 8분`


## ⚙️ 설정 파일

### `configs/config_cloud.yaml`
- 각 폴더 path 및 batch size 설정
- training 파라미터 (epoch, learning rate, early stopping 용 patience 등)
- 모델 활성화 옵션
- GPU 및 출력 설정
- 설정 변수별 상세 내용은 `config_cloud.yaml`주석 확인

## ⚠️ 주의사항

### 실행 환경 요구사항
- **GPU**: H100(sm_90) 또는 A100 GPU 권장
- **CUDA**: 12.1 이상 (H100 sm_90 커널 및 A100 지원)
- **메모리**: 최소 32GB GPU 메모리
- **저장공간**: 
  - MESA 데이터셋 다운로드 시 약 348GB 필요
  - 전처리 후 데이터셋: 32GB
  - 모델 체크포인트: 42MB

---

## 작성자

**SmartM2M AI팀** - 이영우

이 프로젝트는 SleepPPG-Net 기반 수면 단계 분류 모델의 설치 및 실행 가이드를 제공합니다.

> **구현 내용**: 환경 설정, 데이터 준비, 모델 training, 모델 inference, 모니터링, 시각화 모듈 구현
