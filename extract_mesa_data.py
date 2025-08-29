#! /usr/bin/env python3
# (shebang, interpreter 설정)

'''
- extract_mesa_data.py
MESA 데이터셋의 EDF(신호)와 XML(수면 단계 라벨)을 전처리하여 PPG/ECG window 및 수면단계 레이블을 HDF5 파일로 저장하는 모듈

사용 예시 1)
python extract_mesa_data.py

사용 예시 2)
chmod +x extract_mesa_data.py
./extract_mesa_data.py

- 만약 shebang 실행 시 "python3 \r" 오류가 발생한다면,Windows(OS) 환경에서 작성되어 CRLF 줄바꿈이 포함된 경우일 수 있음
아래 명령어로 LF 줄바꿈으로 변환 후 재실행

head -1 extract_mesa_data.py | cat -A
sed -i 's/\r$//' extract_mesa_data.py
./extract_mesa_data.py
'''

import os
import numpy as np
import pandas as pd
import pyedflib
from scipy import signal
from scipy.signal import butter, filtfilt, cheby2
import h5py
from tqdm import tqdm
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
import logging
from logging.handlers import RotatingFileHandler
import traceback

warnings.filterwarnings('ignore')


def setup_logger(log_dir: str, name: str = "MESA"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler (rotating)
    fh = RotatingFileHandler(
        os.path.join(log_dir, "processing.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    ffmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(ffmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    cfmt = logging.Formatter("%(message)s")
    ch.setFormatter(cfmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class MESADataExtractor:
    def __init__(self, mesa_path, output_path, require_ecg=True):
        self.mesa_path = mesa_path
        self.output_path = output_path
        self.require_ecg = require_ecg
        self.target_fs = 34.133333333
        self.window_duration = 30
        self.samples_per_window = 1024
        self.target_hours = 10
        self.target_windows = 1200
        self.target_length = self.target_windows * self.samples_per_window  # 1,228,800

        os.makedirs(output_path, exist_ok=True)
        self.log_dir = os.path.join(self.output_path, "logs")
        self.logger = setup_logger(self.log_dir)

        self.logger.info("=== MESADataExtractor initialized ===")
        self.logger.info(f"require_ecg={self.require_ecg}, target_fs={self.target_fs}, "
                         f"windows={self.target_windows}, samples_per_window={self.samples_per_window}")

    # ────────────────────────────────────────────────────────────────────────────────
    # 1) 시그널 라벨 분석 (옵션)
    # ────────────────────────────────────────────────────────────────────────────────
    def analyze_signal_labels(self, edf_dir):
        self.logger.info("Analyzing signal labels in EDF files...")
        edf_files = [f for f in os.listdir(edf_dir) if f.endswith('.edf')]
        all_labels, ecg_labels, ppg_labels = [], [], []

        for edf_file in tqdm(edf_files[:100], desc="Analyzing signals"):
            try:
                f = pyedflib.EdfReader(os.path.join(edf_dir, edf_file))
                labels = f.getSignalLabels()
                all_labels.extend(labels)
                for label in labels:
                    l = label.lower()
                    if 'ecg' in l or 'ekg' in l:
                        ecg_labels.append(label)
                    if 'pleth' in l or 'ppg' in l:
                        ppg_labels.append(label)
                f.close()
            except Exception as e:
                self.logger.warning(f"[analyze] Failed to read {edf_file}: {e}")
                continue

        label_counts = Counter(all_labels)
        ecg_counts = Counter(ecg_labels)
        ppg_counts = Counter(ppg_labels)

        # 로그 파일로 요약 저장
        self.logger.info("Top 20 most common signal labels:")
        for label, count in label_counts.most_common(20):
            self.logger.info(f"  {label}: {count}")

        self.logger.info("ECG-related labels:")
        for label, count in ecg_counts.most_common():
            self.logger.info(f"  {label}: {count}")

        self.logger.info("PPG-related labels:")
        for label, count in ppg_counts.most_common():
            self.logger.info(f"  {label}: {count}")

    # ────────────────────────────────────────────────────────────────────────────────
    # 2) EDF에서 신호 추출
    # ────────────────────────────────────────────────────────────────────────────────
    def extract_signals_from_edf(self, edf_file):
        try:
            f = pyedflib.EdfReader(edf_file)
            signal_labels = f.getSignalLabels()

            # PPG
            ppg_idx = None
            for idx, label in enumerate(signal_labels):
                if 'pleth' in label.lower() or 'ppg' in label.lower():
                    ppg_idx = idx
                    break

            # ECG (우선순위)
            ecg_idx = None
            ecg_priorities = [
                lambda x: 'ecg' in x and ('ii' in x or '2' in x),
                lambda x: 'ekg' in x and ('ii' in x or '2' in x),
                lambda x: 'ecg' in x,
                lambda x: 'ekg' in x,
                lambda x: 'ekgr' in x,
            ]
            for priority_func in ecg_priorities:
                for idx, label in enumerate(signal_labels):
                    if priority_func(label.lower()):
                        ecg_idx = idx
                        break
                if ecg_idx is not None:
                    break

            if ppg_idx is None:
                self.logger.warning(f"[extract] No PPG in {os.path.basename(edf_file)}")
                f.close()
                return None, None, None, None

            if ecg_idx is None and self.require_ecg:
                self.logger.warning(f"[extract] No ECG in {os.path.basename(edf_file)} (require_ecg=True)")
                f.close()
                return None, None, None, None

            ppg_signal = f.readSignal(ppg_idx)
            ppg_fs = f.getSampleFrequency(ppg_idx)

            if ecg_idx is not None:
                ecg_signal = f.readSignal(ecg_idx)
                ecg_fs = f.getSampleFrequency(ecg_idx)
            else:
                ecg_signal = None
                ecg_fs = None

            f.close()
            return ppg_signal, ecg_signal, ppg_fs, ecg_fs

        except Exception as e:
            self.logger.error(f"[extract] Error reading {os.path.basename(edf_file)}: {e}")
            self.logger.debug(traceback.format_exc())
            return None, None, None, None

    # ────────────────────────────────────────────────────────────────────────────────
    # 3) 전처리 (PPG/ECG)
    # ────────────────────────────────────────────────────────────────────────────────
    def preprocess_ppg(self, ppg_signal, original_fs):
        # Low-pass at 8 Hz (Chebyshev II), zero-phase
        nyq = 0.5 * original_fs
        cutoff = 8 / nyq
        sos = cheby2(N=8, rs=40, Wn=cutoff, btype='lowpass', output='sos')
        filtered = signal.sosfiltfilt(sos, ppg_signal)

        # Resample to target_fs
        duration = len(filtered) / original_fs
        n_samples = int(duration * self.target_fs)
        old_idx = np.linspace(0, len(filtered) - 1, len(filtered))
        new_idx = np.linspace(0, len(filtered) - 1, n_samples)
        resampled = np.interp(new_idx, old_idx, filtered)

        # Clip ±3σ → z-score
        mu, sd = np.mean(resampled), np.std(resampled)
        clipped = np.clip(resampled, mu - 3 * sd, mu + 3 * sd)
        wavppg = (clipped - np.mean(clipped)) / (np.std(clipped) + 1e-8)
        return wavppg

    def preprocess_ecg(self, ecg_signal, original_fs):
        nyq = 0.5 * original_fs
        low = 0.5 / nyq
        high = 40.0 / nyq
        if high >= 1:
            high = 0.99
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, ecg_signal)

        duration = len(filtered) / original_fs
        n_samples = int(duration * self.target_fs)
        old_idx = np.linspace(0, len(filtered) - 1, len(filtered))
        new_idx = np.linspace(0, len(filtered) - 1, n_samples)
        resampled = np.interp(new_idx, old_idx, filtered)

        mu, sd = np.mean(resampled), np.std(resampled)
        clipped = np.clip(resampled, mu - 3 * sd, mu + 3 * sd)
        z = (clipped - np.mean(clipped)) / (np.std(clipped) + 1e-8)
        return z

    # ────────────────────────────────────────────────────────────────────────────────
    # 4) 길이 맞추기
    # ────────────────────────────────────────────────────────────────────────────────
    def pad_or_truncate_signal(self, arr, target_length):
        n = len(arr)
        if n >= target_length:
            return arr[:target_length]
        pad = np.zeros(target_length - n, dtype=arr.dtype)
        return np.concatenate([arr, pad])

    # ────────────────────────────────────────────────────────────────────────────────
    # 5) 라벨 파싱/확장
    # ────────────────────────────────────────────────────────────────────────────────
    def parse_sleep_stages(self, xml_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            scored_events = root.find('.//ScoredEvents')
            if scored_events is None:
                return None

            rows = []
            for event in scored_events.iter('ScoredEvent'):
                etype = event.find('EventType')
                etype = etype.text if etype is not None else None
                if etype != 'Stages|Stages':
                    continue

                concept = event.find('EventConcept').text
                start_time = float(event.find('Start').text)
                duration = float(event.find('Duration').text)

                if 'Wake' in concept:
                    stage = 0
                elif 'Stage 1 sleep' in concept or 'Stage 2 sleep' in concept:
                    stage = 1  # Light
                elif 'Stage 3 sleep' in concept or 'Stage 4 sleep' in concept:
                    stage = 2  # Deep
                elif 'REM sleep' in concept:
                    stage = 3
                else:
                    continue

                rows.append({'Start': start_time, 'Duration': duration, 'Stage': stage})

            if not rows:
                return None
            return pd.DataFrame(rows)

        except Exception as e:
            self.logger.warning(f"[labels] XML parse error {os.path.basename(xml_file)}: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def expand_labels_to_windows(self, df_stages, total_duration):
        # 30s epochs; use ceil for end to reduce off-by-one truncation
        num_windows = int(np.ceil(total_duration / self.window_duration))
        labels = np.full(num_windows, -1, dtype=int)

        for _, row in df_stages.iterrows():
            start_idx = int(np.floor(row['Start'] / self.window_duration))
            end_idx = int(np.ceil((row['Start'] + row['Duration']) / self.window_duration))
            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, len(labels))
            if start_idx < end_idx:
                labels[start_idx:end_idx] = row['Stage']
        return labels

    def pad_or_truncate_labels(self, labels, target_length):
        n = len(labels)
        if n >= target_length:
            return labels[:target_length]
        pad = np.full(target_length - n, -1, dtype=int)
        return np.concatenate([labels, pad])

    # ────────────────────────────────────────────────────────────────────────────────
    # 6) 개별 피험자 처리
    # ────────────────────────────────────────────────────────────────────────────────
    def process_subject(self, edf_file, xml_file):
        basename = os.path.basename(edf_file)
        ppg, ecg, ppg_fs, ecg_fs = self.extract_signals_from_edf(edf_file)
        if ppg is None:
            self.logger.info(f"[subject] Skip {basename} (no PPG or ECG required)")
            return None

        ppg_processed = self.preprocess_ppg(ppg, ppg_fs)

        if ecg is not None:
            ecg_processed = self.preprocess_ecg(ecg, ecg_fs)
            has_ecg = True
        else:
            ecg_processed = np.zeros_like(ppg_processed)
            has_ecg = False

        ppg_final = self.pad_or_truncate_signal(ppg_processed, self.target_length)
        ecg_final = self.pad_or_truncate_signal(ecg_processed, self.target_length)

        if os.path.exists(xml_file):
            df_stages = self.parse_sleep_stages(xml_file)
            if df_stages is not None:
                total_duration = len(ppg_processed) / self.target_fs
                labels = self.expand_labels_to_windows(df_stages, total_duration)
                labels_final = self.pad_or_truncate_labels(labels, self.target_windows)
            else:
                labels_final = np.full(self.target_windows, -1, dtype=int)
        else:
            self.logger.warning(f"[subject] XML not found for {basename}, padding labels with -1")
            labels_final = np.full(self.target_windows, -1, dtype=int)

        try:
            ppg_windows = ppg_final.reshape(self.target_windows, self.samples_per_window)
            ecg_windows = ecg_final.reshape(self.target_windows, self.samples_per_window)
        except Exception as e:
            self.logger.error(f"[subject] Reshape error for {basename}: {e}")
            self.logger.debug(f"ppg_final shape={ppg_final.shape}, expected=({self.target_windows*self.samples_per_window},)")
            return None

        return ppg_windows, ecg_windows, labels_final, has_ecg

    # ────────────────────────────────────────────────────────────────────────────────
    # 7) 전체 처리 루프
    # ────────────────────────────────────────────────────────────────────────────────
    def process_all_subjects(self, subject_list=None, analyze_first=False):
        edf_dir = os.path.join(self.mesa_path, 'polysomnography', 'edfs')
        xml_dir = os.path.join(self.mesa_path, 'polysomnography', 'annotations-events-nsrr')
        self.logger.info(f"EDF dir: {edf_dir}")
        self.logger.info(f"XML dir: {xml_dir}")

        if analyze_first:
            self.analyze_signal_labels(edf_dir)
            return 0

        edf_files = [f for f in os.listdir(edf_dir) if f.endswith('.edf')]
        if subject_list:
            edf_files = [f for f in edf_files if any(subj in f for subj in subject_list)]

        all_ppg_windows, all_ecg_windows, all_labels = [], [], []
        subject_ids, has_real_ecg = [], []

        subjects_with_ecg = 0
        subjects_without_ecg = 0
        failed_subjects = 0

        self.logger.info(f"Start processing subjects: {len(edf_files)} file(s)")
        for edf_filename in tqdm(edf_files, desc="Processing subjects"):
            edf_path = os.path.join(edf_dir, edf_filename)
            # mesa-sleep-0001.edf -> 0001
            try:
                subject_id = edf_filename.split('-')[2].split('.')[0]
            except Exception:
                subject_id = edf_filename.replace('.edf', '')

            xml_filename = edf_filename.replace('.edf', '-nsrr.xml')
            xml_path = os.path.join(xml_dir, xml_filename)

            try:
                result = self.process_subject(edf_path, xml_path)
                if result is None:
                    failed_subjects += 1
                    continue

                ppg_windows, ecg_windows, labels, has_ecg = result
                subjects_with_ecg += int(has_ecg)
                subjects_without_ecg += int(not has_ecg)

                all_ppg_windows.append(ppg_windows)
                all_ecg_windows.append(ecg_windows)
                all_labels.append(labels)
                subject_ids.extend([subject_id] * len(ppg_windows))
                has_real_ecg.extend([has_ecg] * len(ppg_windows))

            except Exception as e:
                failed_subjects += 1
                self.logger.error(f"[loop] Error processing {edf_filename}: {e}")
                self.logger.debug(traceback.format_exc())
                continue

        self.logger.info("\n=== Processing summary ===")
        self.logger.info(f"  Subjects with ECG: {subjects_with_ecg}")
        self.logger.info(f"  Subjects without ECG: {subjects_without_ecg}")
        self.logger.info(f"  Failed subjects: {failed_subjects}")
        self.logger.info(f"  Total subjects attempted: {subjects_with_ecg + subjects_without_ecg + failed_subjects}")

        if not all_ppg_windows:
            self.logger.warning("No data accumulated. Nothing to save.")
            return 0

        all_ppg_windows = np.vstack(all_ppg_windows)
        all_ecg_windows = np.vstack(all_ecg_windows)
        all_labels = np.concatenate(all_labels)
        subject_ids = np.array(subject_ids)
        has_real_ecg = np.array(has_real_ecg)

        self.save_data_separate(all_ppg_windows, all_ecg_windows, all_labels, subject_ids, has_real_ecg)
        return len(all_ppg_windows)

    # ────────────────────────────────────────────────────────────────────────────────
    # 8) 저장 & 통계
    # ────────────────────────────────────────────────────────────────────────────────
    def save_data_separate(self, ppg_windows, ecg_windows, labels, subject_ids, has_real_ecg):
        ppg_file = os.path.join(self.output_path, 'mesa_ppg_with_labels.h5')
        self.logger.info(f"Saving PPG data to {ppg_file}...")
        with h5py.File(ppg_file, 'w') as f:
            f.create_dataset('ppg', data=ppg_windows, compression='gzip',
                             chunks=(100, self.samples_per_window))
            f.create_dataset('labels', data=labels, compression='gzip')
            f.create_dataset('subject_ids', data=subject_ids.astype('S10'), compression='gzip')
            f.attrs['sampling_rate'] = self.target_fs
            f.attrs['window_duration'] = self.window_duration
            f.attrs['samples_per_window'] = self.samples_per_window
            f.attrs['total_windows'] = len(ppg_windows)
            f.attrs['total_subjects'] = len(np.unique(subject_ids))

        ecg_file = os.path.join(self.output_path, 'mesa_real_ecg.h5')
        self.logger.info(f"Saving ECG data to {ecg_file}...")
        with h5py.File(ecg_file, 'w') as f:
            f.create_dataset('ecg', data=ecg_windows, compression='gzip',
                             chunks=(100, self.samples_per_window))
            f.create_dataset('has_real_ecg', data=has_real_ecg, compression='gzip')
            f.create_dataset('subject_ids', data=subject_ids.astype('S10'), compression='gzip')
            f.create_dataset('labels', data=labels, compression='gzip')
            real_ecg_indices = np.where(has_real_ecg)[0]
            f.create_dataset('real_ecg_indices', data=real_ecg_indices, compression='gzip')
            f.attrs['windows_with_real_ecg'] = int(np.sum(has_real_ecg))
            f.attrs['windows_without_real_ecg'] = int(np.sum(~has_real_ecg))

        index_file = os.path.join(self.output_path, 'mesa_subject_index.h5')
        self.logger.info(f"Creating index file {index_file}...")
        with h5py.File(index_file, 'w') as f:
            unique_subjects = np.unique(subject_ids)
            subject_group = f.create_group('subjects')

            for subj in unique_subjects:
                subj_str = subj.decode() if isinstance(subj, bytes) else str(subj)
                mask = (subject_ids == subj)
                indices = np.where(mask)[0]

                subj_group = subject_group.create_group(subj_str)
                subj_group.create_dataset('window_indices', data=indices)
                subj_group.attrs['n_windows'] = len(indices)
                subj_group.attrs['has_ecg'] = bool(has_real_ecg[indices[0]])

            f.attrs['total_subjects'] = len(unique_subjects)
            f.attrs['total_windows'] = len(ppg_windows)

        self.save_statistics(ppg_windows, ecg_windows, labels, subject_ids, has_real_ecg)
        self.logger.info("Data saved successfully.")

    def save_statistics(self, ppg_windows, ecg_windows, labels, subject_ids, has_real_ecg):
        valid_labels = labels[labels != -1]
        stats = {
            'total_windows': len(ppg_windows),
            'total_subjects': len(np.unique(subject_ids)),
            'windows_with_real_ecg': int(np.sum(has_real_ecg)),
            'windows_without_real_ecg': int(np.sum(~has_real_ecg)),
            'ppg_shape': ppg_windows.shape,
            'ecg_shape': ecg_windows.shape,
            'valid_labels': len(valid_labels),
            'label_distribution': dict(zip(*np.unique(valid_labels, return_counts=True))) if len(valid_labels) > 0 else {},
            'sampling_rate': self.target_fs,
            'window_duration': self.window_duration,
            'samples_per_window': self.samples_per_window,
            'file_structure': {
                'mesa_ppg_with_labels.h5': ['ppg', 'labels', 'subject_ids'],
                'mesa_real_ecg.h5': ['ecg', 'has_real_ecg', 'subject_ids', 'labels', 'real_ecg_indices'],
                'mesa_subject_index.h5': ['subjects/{subject_id}/window_indices']
            }
        }

        stats_file = os.path.join(self.output_path, 'data_stats.npy')
        np.save(stats_file, stats)

        stats_txt = os.path.join(self.output_path, 'data_stats.txt')
        with open(stats_txt, 'w', encoding='utf-8') as f:
            f.write("MESA Sleep Data Processing Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total windows: {stats['total_windows']}\n")
            f.write(f"Total subjects: {stats['total_subjects']}\n")
            f.write(f"Windows with real ECG: {stats['windows_with_real_ecg']}\n")
            f.write(f"Windows without real ECG: {stats['windows_without_real_ecg']}\n")
            f.write(f"Valid labels: {stats['valid_labels']}\n")
            f.write(f"\nLabel distribution:\n")
            stage_names = {0: 'Wake', 1: 'Light', 2: 'Deep', 3: 'REM'}
            for label, count in stats['label_distribution'].items():
                f.write(f"  {stage_names.get(int(label), f'Stage{label}')}: {count}\n")
            f.write(f"\nSampling rate: {stats['sampling_rate']} Hz\n")
            f.write(f"Window duration: {stats['window_duration']} seconds\n")
            f.write(f"Samples per window: {stats['samples_per_window']}\n")

        self.logger.info(f"Stats saved: {stats_file}, {stats_txt}")


def main():
    MESA_PATH = "./mesa-commercial-use" # 원본 MESA 데이터 루트
    # MESA_PATH = "./mesa" # 원본 MESA 데이터 루트
    OUTPUT_PATH = "./mesa-x" # 전처리 출력 폴더

    extractor = MESADataExtractor(MESA_PATH, OUTPUT_PATH, require_ecg=False)

    # 1) 신호 라벨 목록만 먼저 보고 싶으면:
    # extractor.process_all_subjects(analyze_first=True); return

    # 2) 특정 피험자만 처리하고 싶으면:
    # n_windows = extractor.process_all_subjects(subject_list=['0001','0002'])

    # 3) 전체 처리:
    n_windows = extractor.process_all_subjects()
    print(f"\nProcessing completed! Total windows: {n_windows}")
    # 로그는 OUTPUT_PATH/logs/processing.log에 쌓임


if __name__ == "__main__":
    main()
