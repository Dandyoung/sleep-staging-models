#!/usr/bin/env python3 
# (shebang, interpreter 설정)

'''
- verify_mesa_outputs.py
전처리된 MESA 데이터셋 검증 (1 ~ 4단계 + input 길이(window들의 전체 길이) 확인)하는 모듈

- 사용 예시 1)
python verify_mesa_outputs.py --base ./mesa-x

- 사용 예시 2)
chmod +x verify_mesa_dataset.py
./verify_mesa_outputs.py --base ./mesa-x
'''

import os
import argparse
import numpy as np
import h5py
from scipy.signal import welch

# 기대되는 값 (검증 기준)
TARGET_FS = 34.133333333           # 목표 샘플링레이트 (Hz)
WINDOW_DURATION = 30               # 윈도우 길이 (초)
SAMPLES_PER_WINDOW = 1024          # 윈도우당 샘플 수
WINDOWS_PER_SUBJECT = 1200         # 피험자당 윈도우 개수
TOTAL_SAMPLES = WINDOWS_PER_SUBJECT * SAMPLES_PER_WINDOW  # 총 샘플 수 (1,228,800)

# 라벨 → 단계 이름 매핑
STAGE_NAMES = {-1: "패딩(-1)", 0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}

def compute_label_distribution(labels):
    """라벨 분포 계산"""
    uniq, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}

def format_label_distribution(dist):
    """라벨 분포를 사람이 읽기 쉽게 문자열로 변환"""
    return ", ".join([f"{STAGE_NAMES.get(k, str(k))}:{v}" for k, v in sorted(dist.items())])

def check_ppg(ppg_h5):
    """PPG 데이터(h5) 핵심 검증"""
    with h5py.File(ppg_h5, "r") as f:
        ppg = f["ppg"][:]                 # (N, 1024)
        labels = f["labels"][:]           # (N,)
        subject_ids = f["subject_ids"][:] # (N,)
        attrs = dict(f.attrs)

    N, W = ppg.shape
    print(f"[PPG] 전체 윈도우 개수={N}, 윈도우당 샘플 수={W}")
    print(f"[PPG] 메타정보: 샘플링레이트={attrs.get('sampling_rate')}, "
          f"윈도우 길이(초)={attrs.get('window_duration')}, "
          f"윈도우 샘플수={attrs.get('samples_per_window')}")

    # (1) 윈도우 크기 검증
    assert W == SAMPLES_PER_WINDOW, "PPG: 윈도우 샘플 수가 1024가 아님"
    assert int(attrs.get("window_duration", WINDOW_DURATION)) == WINDOW_DURATION, "PPG: 윈도우 길이(30초) 불일치"
    assert int(attrs.get("samples_per_window", SAMPLES_PER_WINDOW)) == SAMPLES_PER_WINDOW, "PPG: 샘플수 불일치"

    # (2) 다운샘플링 검증 → fs * 30 ≈ 1024
    fs = float(attrs.get("sampling_rate", TARGET_FS))
    calc = fs * WINDOW_DURATION
    assert abs(calc - SAMPLES_PER_WINDOW) < 1e-6, f"다운샘플링 불일치: {calc} ≠ 1024"

    # (3) 라벨 값 범위 검증
    assert set(np.unique(labels)).issubset({-1,0,1,2,3}), "라벨 값이 허용 범위 초과"

    # (4) 라벨 분포 출력
    dist = compute_label_distribution(labels)
    print("[PPG] 라벨 분포:", format_label_distribution(dist))

    # (5) 피험자별 윈도우 수와 총 길이 검증
    unique_subjects = np.unique(subject_ids)
    success_count = 0
    for u in unique_subjects:
        idx = np.where(subject_ids == u)[0]
        if len(idx) == 0: 
            continue
        if len(idx) != WINDOWS_PER_SUBJECT:
            raise AssertionError(f"피험자 {u!r}: 윈도우 수 {len(idx)} ≠ {WINDOWS_PER_SUBJECT}")
        if ppg[idx].size != TOTAL_SAMPLES:
            raise AssertionError(f"피험자 {u!r}: 총 샘플 수 {ppg[idx].size} ≠ {TOTAL_SAMPLES}")
        success_count += 1
    print(f"[PPG] 모든 피험자별 윈도우 수={WINDOWS_PER_SUBJECT}, 총 샘플 수={TOTAL_SAMPLES} 확인 완료 ({success_count}명)")

    return ppg, labels, fs

def check_filter(ppg, fs):
    """8Hz 저역통과 필터 적용 여부 간단 검증"""
    valid_idx = np.where(ppg.std(axis=1) > 0)[0]
    if len(valid_idx) == 0:
        print("[필터] 유효 윈도우 없음")
        return
    sample = ppg[valid_idx[:10]].mean(axis=0)  # 앞쪽 10개 평균 파형
    f, Pxx = welch(sample, fs=fs, nperseg=512)
    ratio = np.sum(Pxx[f > 8]) / np.sum(Pxx)
    print(f"[필터] 8Hz 이상 주파수 에너지 비율 ≈ {ratio:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="전처리 산출물 폴더 경로")
    args = ap.parse_args()
    base = args.base

    ppg_h5 = os.path.join(base, "mesa_ppg_with_labels.h5")

    print(f"[검증 시작] 경로: {base}")
    ppg, labels, fs = check_ppg(ppg_h5)
    check_filter(ppg, fs)

    print("\n전처리 데이터 검증 완료 (1~4단계 + 입력 길이 확인)")

if __name__ == "__main__":
    main()
