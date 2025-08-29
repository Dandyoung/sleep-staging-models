import torch

ckpt = torch.load("/workspace/NSRR/sleep-staging-models/models/best_model.pth", map_location="cpu")

print(ckpt.keys())  # 저장된 키 확인

# epoch 번호 확인
print("Epoch:", ckpt['epoch'])
print("Best validation loss:", ckpt['best_validation_loss'])
print("Best kappa:", ckpt['best_overall_kappa'])

# 모델 파라미터 키 몇 개만 미리보기
print("Model state_dict keys:", list(ckpt['model_state_dict'].keys())[:10])
