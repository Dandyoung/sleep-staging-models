# Sleep Staging Models - PPG 기반 수면 단계 분류 모델(SleepPPG-Net)

> **참고**: 이 프로젝트는 [DavyWJW/sleep-staging-models](https://github.com/DavyWJW/sleep-staging-models/tree/main?tab=readme-ov-file) GitHub 레포지토리를 기반으로 합니다.

## 🚀 빠른 시작

### 1. 환경 설정

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

## 🏗️ 프로젝트 구조

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
│   ├── mesa_ppg_with_labels.h5    # PPG 데이터 + 라벨 (15GB)
│   ├── mesa_real_ecg.h5           # 실제 ECG 데이터 (16GB)
│   ├── mesa_subject_index.h5      # 피실험자 인덱스 정보 (20MB)
│   ├── data_stats.npy             # 데이터 통계 (numpy)
│   ├── data_stats.txt             # 데이터 통계 (텍스트)
│   └── logs/                      # 전처리 로그
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

## 📊 주요 파일 설명

### 핵심 모델 파일
- **`multimodal_sleep_model.py`**: 주요 모델 아키텍처 (SleepPPGNet, MultiModalSleepNet)
- **`train_ppg_only.py`**: PPG 전용 모델 training 스크립트
- **`inference_ppg_only.py`**: PPG 전용 모델 inference 스크립트
- **`extract_mesa_data.py`**: MESA 데이터 추출 및 전처리

### 데이터 처리
- **`multimodal_dataset_aligned.py`**: 데이터셋 로더 및 배치 생성
- **`verify_mesa_dataset.py`**: EDF/XML 파일 1:1 매핑 검증
- **`verify_mesa_outputs.py`**: 전처리된 데이터셋 및 입력 길이 검증

### 데이터 검증 및 전처리
- **`verify_mesa_dataset.py`**: 다운로드된 MESA 데이터셋 검증
- **`verify_mesa_outputs.py`**: 전처리 결과 및 window 길이 확인
- **`extract_mesa_data.py`**: SleepPPG-Net 학습용 데이터 전처리

## 📈 데이터셋 크기 정보

| 항목 | 크기 | 설명 |
|------|------|------|
| **전체 데이터셋** | **348GB** | MESA 데이터셋 전체 크기 |
| **XML 파일** | **444MB** | 수면 단계 어노테이션 파일들 |
| **EDF 파일** | **347GB** | 생체신호 데이터 파일들 |
| **전처리 후 데이터셋** | **32GB** | 모델 학습용으로 가공된 데이터 |

### 데이터셋 상세 정보
- **총 피실험자 수**: 1,900명
- **EDF/XML 매칭**: 1,900개 (100% 매칭)
- **전처리 후 파일**:
  - `mesa_ppg_with_labels.h5`: 15GB (PPG 신호 + 라벨)
  - `mesa_real_ecg.h5`: 16GB (ECG 신호)
  - `mesa_subject_index.h5`: 20MB (피실험자 인덱스)

## ⚙️ 설정 파일

### `configs/config_cloud.yaml`
- 데이터 경로 및 배치 크기 설정
- training 파라미터 (에포크, 학습률, 조기 종료)
- 모델 활성화 옵션
- GPU 및 출력 설정

## 📦 의존성 패키지

### 핵심 프레임워크
- **PyTorch 2.5.1+ (CUDA 12.1)**: 딥러닝 프레임워크 (H100/A100 지원)
- **NumPy 1.24.3**: 수치 계산
- **scikit-learn 1.3.0**: 머신러닝 유틸리티

> **GPU 호환성**: H100(sm_90) 및 A100 GPU 사용 시 CUDA 12.1 버전 PyTorch 필수

### 신호 처리
- **neurokit2 0.2.5**: 생체신호 처리
- **biosppy 2.1.1**: 생체신호 분석
- **pyedflib 0.1.34**: EDF 파일 처리
- **wfdb 4.1.2**: PhysioNet 데이터베이스

### 시각화 및 모니터링
- **matplotlib 3.7.2**: 그래프 생성
- **tensorboard 2.13.0**: training 모니터링
- **wandb 0.15.8**: 실험 추적

## ⚠️ 주의사항

### 실행 환경 요구사항
- **GPU**: H100(sm_90) 또는 A100 GPU 권장
- **CUDA**: 12.1 이상 (sm_90 커널 지원)
- **메모리**: 최소 32GB GPU 메모리
- **저장공간**: 
  - MESA 데이터셋 다운로드 시 약 348GB 필요
  - 전처리 후 데이터셋: 32GB
  - 모델 체크포인트: 42MB

### 문제 해결
- **CUDA 에러**: `no kernel image is available for execution on device` 발생 시 CUDA 12.1 버전 PyTorch 재설치
- **메모리 부족**: 배치 크기 조정 또는 GPU 메모리 확인
- **저장공간 부족**: 전처리 후 원본 데이터 삭제 고려 (32GB vs 348GB)

---

## 👨‍💻 작성자

**SmartM2M AI팀** - 이영우

이 프로젝트는 PPG 기반 수면 단계 분류 모델의 설치 및 실행 가이드를 제공합니다.

> **구현 내용**: 환경 설정, 데이터 준비, 모델 training, 모델 inference, 모니터링 등 전체 파이프라인 구현
