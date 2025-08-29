'''
- 학습 실행 스크립트
1. fg : python train_ppg_only.py --config configs/config_cloud.yaml --model ppg_only
2. bg : nohup python train_ppg_only.py --config configs/config_cloud.yaml --model ppg_only &

- TensorBoard 실행 스크립트
0. 파일 읽기/디렉토리 접근 권한 부여 : chmod -R a+rX /outputs
1. fg : tensorboard --logdir outputs --host 0.0.0.0 --port 8890
2. bg : nohup tensorboard --logdir outputs --host 0.0.0.0 --port 8890 &

- 실행 후 PID 확인
ps -ef | grep train_ppg_only.py | grep -v grep
ps -ef | grep tensorboard | grep -v grep
# 또는
pgrep -fl train_ppg_only.py
pgrep -fl tensorboard

- 종료시
kill <PID>
# 또는
pkill -f train_ppg_only.py
pkill -f tensorboard
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, f1_score
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import gc
import argparse
import yaml

# === 추가: 콘솔 로깅용 ===
import sys
import atexit
from pathlib import Path

from multimodal_sleep_model import SleepPPGNet, MultiModalSleepNet
from multimodal_dataset_aligned import get_dataloaders


class MultiModalTrainer:
    def __init__(self, config, run_id=None):
        self.config = config
        self.run_id = run_id
        self.device = torch.device(
            f'cuda:{config["gpu"]["device_id"]}' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")

        # 출력/로그/체크포인트 디렉토리 준비
        self.setup_directories()

        # TensorBoard 작성기
        self.writer = SummaryWriter(self.log_dir)

        # 사용한 설정 저장
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # === 콘솔 출력 tee 설정 (print / tqdm 모두 파일로 복제) ===
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        # line-buffered (buffering=1), utf-8
        self._console_fp = open(self.console_log_path, mode='a', buffering=1, encoding='utf-8')

        class _Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    try:
                        s.write(data)
                    except Exception:
                        pass
            def flush(self):
                for s in self.streams:
                    try:
                        s.flush()
                    except Exception:
                        pass

        sys.stdout = _Tee(sys.stdout, self._console_fp)
        sys.stderr = _Tee(sys.stderr, self._console_fp)
        print(f"[Console Tee] Logging to {self.console_log_path}")

        # 비정상 종료 대비 안전 복구
        def _cleanup():
            try:
                sys.stdout.flush(); sys.stderr.flush()
            except Exception:
                pass
            try:
                sys.stdout = self._orig_stdout
                sys.stderr = self._orig_stderr
            except Exception:
                pass
            try:
                if not self._console_fp.closed:
                    self._console_fp.close()
            except Exception:
                pass
        atexit.register(_cleanup)
        self._cleanup = _cleanup  # 필요 시 수동 호출

    def setup_directories(self):
        """output / checkpoints / logs / results / console_logs 디렉토리 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{self.config['model_type']}_{timestamp}"
        if self.run_id is not None:
            model_name += f"_run{self.run_id}"

        self.output_dir = os.path.join(self.config['output']['save_dir'], model_name)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')            # TensorBoard
        self.results_dir = os.path.join(self.output_dir, 'results')
        self.console_log_dir = os.path.join(self.output_dir, 'console_logs')  # ★ 콘솔 로그

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.console_log_dir, exist_ok=True)

        # 콘솔 로그 파일 경로
        self.console_log_path = os.path.join(self.console_log_dir, f'{Path(self.output_dir).name}.log')

    def create_model(self):
        """모델 생성"""
        if self.config['model_type'] == 'ppg_only':
            model = SleepPPGNet()
        elif self.config['model_type'] == 'ppg_with_noise':
            from ppg_with_noise_baseline import PPGWithNoiseBaseline
            model = PPGWithNoiseBaseline(
                noise_config=self.config.get('noise_config', None)
            )
        elif self.config['model_type'] == 'multimodal':
            model = MultiModalSleepNet(
                fusion_strategy=self.config.get('fusion_strategy', 'attention')
            )
        else:
            raise ValueError(f"Unknown model_type: {self.config['model_type']}")
        return model.to(self.device)

    def calculate_class_weights(self, train_dataset):
        """클래스별 샘플 수에 따라 가중치 계산"""
        print("\n클래스 가중치 계산 중...")

        all_labels = []
        sample_size = min(len(train_dataset), 50)  # 최대 50명만 샘플링해서 분포 추정

        for idx in tqdm(range(sample_size), desc="라벨 샘플링", dynamic_ncols=True, mininterval=0.3):
            data = train_dataset[idx]
            if len(data) == 2:   # (ppg, labels)
                _, labels = data
            else:                # (ppg, ecg, labels)
                _, _, labels = data

            valid_labels = labels[labels != -1]
            all_labels.extend(valid_labels.numpy().tolist())

        # 라벨별 개수 카운트
        label_counts = Counter(all_labels)
        class_counts = [label_counts.get(i, 1) for i in range(4)]  # 최소 1로 보정
        total_samples = sum(class_counts)

        # 분포 출력
        print(f"\n샘플링 {sample_size}명 기준 라벨 분포:")
        stage_names = ['Wake', 'Light', 'Deep', 'REM']
        for i, count in enumerate(class_counts):
            pct = count / total_samples * 100
            print(f"  {stage_names[i]} (class {i}): {count}개 ({pct:.2f}%)")

        # 클래스 샘플 수(count)에 따라 가중치 설정
        class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        print(f"\n클래스 가중치: {class_weights}")
        return class_weights.to(self.device)

    def train_epoch(self, dataloader, model, device, optimizer, criterion):
        """1 에폭 학습"""
        model.train()
        running_loss = 0.0
        total = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", dynamic_ncols=True, mininterval=0.3)):
            if len(batch) == 2:
                ppg, labels = batch
                ppg = ppg.to(device)
                labels = labels.to(device)
                outputs = model(ppg)
            else:
                ppg, ecg, labels = batch
                ppg = ppg.to(device)
                ecg = ecg.to(device)
                labels = labels.to(device)
                outputs = model(ppg, ecg)

            optimizer.zero_grad()

            # 이중 softmax 확인
            if batch_idx == 0:
                raw_sum_mean  = outputs.sum(dim=1).mean().item()
                soft_sum_mean = outputs.softmax(dim=1).sum(dim=1).mean().item()
                print(f"[softmax-debug/train] raw_sum_mean={raw_sum_mean:.4f}, softmax_sum_mean={soft_sum_mean:.4f}")

            # (B, 4, 1200) → (B, 1200, 4)
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                labels.reshape(-1)
            ).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 그래디언트 클리핑
            optimizer.step()

            # -1 제외하고 유효 라벨만 집계
            mask = labels != -1
            valid_labels = labels[mask]
            total += valid_labels.size(0)
            running_loss += loss.item() * valid_labels.size(0)

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()
        return running_loss / total if total > 0 else 0

    def validate(self, dataloader, model, device, criterion):
        """검증 (accuracy, f1, kappa 등 계산)"""
        model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        patient_predictions, patient_labels = defaultdict(list), defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", dynamic_ncols=True, mininterval=0.3)):
                if len(batch) == 2:
                    ppg, labels = batch
                    ppg, labels = ppg.to(device), labels.to(device)
                    outputs = model(ppg)
                else:
                    ppg, ecg, labels = batch
                    ppg, ecg, labels = ppg.to(device), ecg.to(device), labels.to(device)
                    outputs = model(ppg, ecg)

                # 이중 softmax 확인
                if batch_idx == 0:
                    raw_sum_mean  = outputs.sum(dim=1).mean().item()
                    soft_sum_mean = outputs.softmax(dim=1).sum(dim=1).mean().item()
                    print(f"[softmax-debug/train] raw_sum_mean={raw_sum_mean:.4f}, softmax_sum_mean={soft_sum_mean:.4f}")

                outputs = outputs.permute(0, 2, 1)  # (B, 1200, 4)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    labels.reshape(-1)
                ).mean()

                # 환자 단위로 결과 저장
                batch_size = outputs.shape[0]
                for i in range(batch_size):
                    patient_idx = batch_idx * dataloader.batch_size + i
                    mask = labels[i] != -1
                    if mask.any():
                        patient_outputs = outputs[i][mask]
                        patient_labels_i = labels[i][mask]
                        _, predicted = patient_outputs.max(1)

                        patient_predictions[patient_idx].extend(predicted.cpu().numpy())
                        patient_labels[patient_idx].extend(patient_labels_i.cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(patient_labels_i.cpu().numpy())
                        running_loss += loss.item() * patient_labels_i.numel()

        # 환자별 kappa
        patient_kappas = [
            cohen_kappa_score(patient_labels[idx], patient_predictions[idx])
            for idx in patient_predictions if len(np.unique(patient_labels[idx])) > 1
        ]

        # 전체 지표
        overall_acc = np.mean(np.array(all_preds) == np.array(all_labels)) if all_labels else 0
        overall_kappa = cohen_kappa_score(all_labels, all_preds) if all_labels else 0
        overall_f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0
        median_kappa = np.median(patient_kappas) if patient_kappas else 0

        # confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'loss': running_loss / len(all_labels) if all_labels else 0,
            'overall_accuracy': overall_acc,
            'overall_kappa': overall_kappa,
            'overall_f1': overall_f1,
            'median_kappa': median_kappa,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'patient_kappas': patient_kappas,
            'confusion_matrix': cm
        }

    def calculate_per_class_metrics(self, cm):
        """클래스별 precision/recall/f1 계산"""
        n_classes = cm.shape[0]
        precision, recall, f1 = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def plot_confusion_matrix(self, cm, epoch):
        """confusion matrix 시각화"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Wake', 'Light', 'Deep', 'REM'],
                    yticklabels=['Wake', 'Light', 'Deep', 'REM'])
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        save_path = os.path.join(self.results_dir, f'confusion_matrix_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()

    def train(self):
        """메인 학습 루프"""
        print(f"\n{'=' * 60}")
        print(f"Training {self.config['model_type']} model")
        if self.run_id is not None:
            print(f"Run {self.run_id}/{self.config['training']['num_runs']}")
        print(f"{'=' * 60}")

        # 데이터 경로 준비
        data_paths = {
            'ppg': self.config['data']['ppg_file'],
            'index': self.config['data']['index_file']
        }

        # 데이터로더 생성
        print(f"\nLoading data...")
        if self.config['model_type'] in ['ppg_only', 'ppg_with_noise']:
            train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
                data_paths,
                batch_size=self.config['data']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                model_type='ppg_only',
                use_sleepppg_test_set=self.config['training']['use_sleepppg_test_set']
            )
        else:
            data_paths['real_ecg'] = self.config['data']['ecg_file']
            train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
                data_paths,
                batch_size=self.config['data']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                model_type='multimodal',
                use_generated_ecg=False,
                use_sleepppg_test_set=self.config['training']['use_sleepppg_test_set']
            )

        # 모델 생성
        model = self.create_model()
        print(f"Model created on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # 클래스 가중치 계산
        class_weights = self.calculate_class_weights(train_dataset)

        # 손실 함수/옵티마이저
        train_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, reduction="none")
        val_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # 학습률 스케줄러 (overall kappa 기준)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=5, verbose=True
        )

        # 베스트 모델 저장용
        best_validation_loss = float('inf')
        best_validation_kappa = 0
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')

        patience = self.config['training']['patience']
        trigger_times = 0

        training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_overall_accuracies': [],
            'val_overall_kappas': [],
            'val_overall_f1_scores': [],
        }

        num_epochs = self.config['training']['num_epochs']
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print('=' * 50)

            # 학습
            train_loss = self.train_epoch(
                train_loader, model, self.device, optimizer, train_criterion
            )
            training_history['train_losses'].append(train_loss)
            print(f"Training Loss: {train_loss:.4f}")

            # 검증
            val_results = self.validate(val_loader, model, self.device, val_criterion)

            # 기록
            training_history['val_losses'].append(val_results['loss'])
            training_history['val_overall_accuracies'].append(val_results['overall_accuracy'])
            training_history['val_overall_kappas'].append(val_results['overall_kappa'])
            training_history['val_overall_f1_scores'].append(val_results['overall_f1'])

            print(f"\nValidation Results:")
            print(f"  Loss: {val_results['loss']:.4f}")
            print(f"  Overall - Acc: {val_results['overall_accuracy']:.4f}, "
                  f"Kappa: {val_results['overall_kappa']:.4f}, F1: {val_results['overall_f1']:.4f}")

            # 텐서보드
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_results['loss'], epoch)
            self.writer.add_scalar('Val/Overall_Accuracy', val_results['overall_accuracy'], epoch)
            self.writer.add_scalar('Val/Overall_Kappa', val_results['overall_kappa'], epoch)
            self.writer.add_scalar('Val/Overall_F1', val_results['overall_f1'], epoch)

            # 스케줄러 업데이트 (kappa 기준)
            scheduler.step(val_results['overall_kappa'])

            # 베스트 업데이트: 우선 kappa 확인, 그 다음 loss
            if val_results['overall_kappa'] > best_validation_kappa:
                best_validation_kappa = val_results['overall_kappa']

            if val_results['loss'] < best_validation_loss:
                best_validation_loss = val_results['loss']
                trigger_times = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_validation_loss': best_validation_loss,
                    'best_overall_kappa': val_results['overall_kappa'],
                    'training_history': training_history,
                    'config': self.config
                }
                torch.save(checkpoint, best_model_path)
                print('Saved best model!')

                # 중간 confusion matrix 저장
                if self.config['output']['save_intermediate']:
                    self.plot_confusion_matrix(val_results['confusion_matrix'], epoch)
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            # 주기적 체크포인트 (옵션)
            if epoch % self.config['output']['save_frequency'] == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_history': training_history
                }, checkpoint_path)

        print("\nTraining completed!")

        # 테스트(베스트 모델 로드)
        print("\n" + "=" * 60)
        print("Testing best model on SleepPPG-Net test set...")
        print("=" * 60)

        checkpoint = torch.load(best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_results = self.validate(test_loader, model, self.device, val_criterion)

        print(f"\nTest Results:")
        print(f"  Loss: {test_results['loss']:.4f}")
        print(f"  Overall - Acc: {test_results['overall_accuracy']:.4f}, "
              f"Kappa: {test_results['overall_kappa']:.4f}, F1: {test_results['overall_f1']:.4f}")

        # 세부 리포트
        report_text = classification_report(
            test_results['all_labels'], test_results['all_preds'],
            target_names=['Wake', 'Light', 'Deep', 'REM']
        )
        report = classification_report(
            test_results['all_labels'], test_results['all_preds'],
            target_names=['Wake', 'Light', 'Deep', 'REM'],
            output_dict=True
        )
        print("\nClassification Report:")
        print(report_text)

        # 최종 confusion matrix 저장
        self.plot_confusion_matrix(test_results['confusion_matrix'], 'final')

        # 결과 JSON 저장
        results = {
            'model_type': self.config['model_type'],
            'run_id': self.run_id,
            'test_loss': test_results['loss'],
            'test_accuracy_overall': test_results['overall_accuracy'],
            'test_kappa_overall': test_results['overall_kappa'],
            'test_f1_overall': test_results['overall_f1'],
            'classification_report': report,
            'confusion_matrix': test_results['confusion_matrix'].tolist(),
            'training_history': training_history,
            'best_epoch': checkpoint.get('epoch', None),
            'patient_kappa_stats': {
                'min': float(np.min(test_results['patient_kappas'])) if test_results['patient_kappas'] else None,
                'max': float(np.max(test_results['patient_kappas'])) if test_results['patient_kappas'] else None,
                'mean': float(np.mean(test_results['patient_kappas'])) if test_results['patient_kappas'] else None,
                'std': float(np.std(test_results['patient_kappas'])) if test_results['patient_kappas'] else None,
                'median': float(np.median(test_results['patient_kappas'])) if test_results['patient_kappas'] else None,
                '25_percentile': float(np.percentile(test_results['patient_kappas'], 25)) if test_results['patient_kappas'] else None,
                '75_percentile': float(np.percentile(test_results['patient_kappas'], 75)) if test_results['patient_kappas'] else None
            }
        }

        results_path = os.path.join(self.results_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        self.writer.close()

        # === 콘솔 tee 해제 및 파일 닫기 ===
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except Exception:
            pass
        try:
            if not self._console_fp.closed:
                self._console_fp.close()
        except Exception:
            pass

        return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train sleep staging models')
    parser.add_argument('--config', type=str, default='config_cloud.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['ppg_only', 'ppg_with_noise', 'real_ecg', 'both'],
                        default='both', help='Which model(s) to train')
    parser.add_argument('--runs', type=int, default=None,
                        help='Number of runs (overrides config)')
    return parser.parse_args()


def main():
    """메인 함수: 다회 학습 및 집계 지원"""
    args = parse_args()

    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 커맨드라인 인자 우선
    if args.runs is not None:
        config['training']['num_runs'] = args.runs

    num_runs = config['training'].get('num_runs', 1)

    # 어떤 모델을 학습할지 결정
    models_to_train = []
    if args.model == 'both':
        if config.get('model', {}).get('ppg_only', {}).get('enabled', False):
            models_to_train.append('ppg_only')
        if config.get('model', {}).get('ppg_with_noise', {}).get('enabled', False):
            models_to_train.append('ppg_with_noise')
        if config.get('model', {}).get('real_ecg', {}).get('enabled', False):
            models_to_train.append('real_ecg')
    else:
        models_to_train = [args.model]

    all_results = defaultdict(list)

    for run in range(1, num_runs + 1):
        print(f"\n{'=' * 80}")
        print(f"RUN {run}/{num_runs}")
        print('=' * 80)

        # 시드 고정(런마다 다르게)
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run)

        for model_type in models_to_train:
            print(f"\n{'=' * 60}")
            print(f"Training {model_type} model")
            print('=' * 60)

            # 모델별 설정 복사/보정
            model_config = config.copy()
            model_config['model_type'] = model_type
            if model_type == 'real_ecg':
                model_config['fusion_strategy'] = config['model']['real_ecg']['fusion_strategy']

            # 학습 실행
            trainer = MultiModalTrainer(model_config, run_id=run if num_runs > 1 else None)
            results = trainer.train()
            all_results[model_type].append(results)

            # 메모리 청소
            torch.cuda.empty_cache()
            gc.collect()

    # 여러 번 돌렸다면 간단 집계 출력
    if num_runs > 1:
        print("\n" + "=" * 80)
        print(f"FINAL RESULTS ({num_runs} runs)")
        print("=" * 80)

        for model_type in models_to_train:
            if not all_results[model_type]:
                continue

            overall_accs = [r['test_accuracy_overall'] for r in all_results[model_type]]
            overall_kappas = [r['test_kappa_overall'] for r in all_results[model_type]]
            overall_f1s = [r['test_f1_overall'] for r in all_results[model_type]]

            print(f"\n{model_type.upper()} Model:")
            print(f"  Acc  : median {np.median(overall_accs):.4f} | mean {np.mean(overall_accs):.4f} ± {np.std(overall_accs):.4f}")
            print(f"  Kappa: median {np.median(overall_kappas):.4f} | mean {np.mean(overall_kappas):.4f} ± {np.std(overall_kappas):.4f}")
            print(f"  F1   : median {np.median(overall_f1s):.4f} | mean {np.mean(overall_f1s):.4f} ± {np.std(overall_f1s):.4f}")

        # 요약 저장
        summary = {'num_runs': num_runs, 'results': all_results}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(config['output']['save_dir'], exist_ok=True)
        summary_path = os.path.join(config['output']['save_dir'], f'summary_results_{ts}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary results saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
