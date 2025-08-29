import torch
import torch.nn.functional as F
import torch.nn as nn

# 예시 logits (모델 출력 전 값)
logits = torch.tensor([[1.2, -0.3, 0.7, -2.0]])
target = torch.tensor([1])  # 정답 클래스 인덱스

# 1) Case 1: logits 그대로 입력 (정상)
loss_logits = nn.CrossEntropyLoss()(logits, target)

# 2) Case 2: softmax(logits) 입력
probs1 = F.softmax(logits, dim=1)
loss_probs1 = nn.CrossEntropyLoss()(probs1, target)

# 3) Case 3: softmax(softmax(logits)) 입력
probs2 = F.softmax(probs1, dim=1)
loss_probs2 = nn.CrossEntropyLoss()(probs2, target)

print("=== 입력 값 비교 ===")
print("Raw logits:              ", logits)
print("Softmax(logits):         ", probs1)
print("Softmax(Softmax(logits)):", probs2)

print("\n=== CrossEntropyLoss 결과 ===")
print("Case 1) logits 그대로 입력              →", loss_logits.item())
print("Case 2) softmax(logits) 입력            →", loss_probs1.item())
print("Case 3) softmax(softmax(logits)) 입력   →", loss_probs2.item())