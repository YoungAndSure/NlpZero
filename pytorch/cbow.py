#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.util import *
import matplotlib.pyplot as plt
import torch
from torch import nn
from easy_data import EasyDataset
from torch.utils.data import DataLoader

train_dataset = EasyDataset()
test_dataset = EasyDataset()

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# 初始化模型
class CbowModel(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super().__init__()
    self.in_emb = nn.Embedding(vocab_size, hidden_size)
    self.out_emb = nn.Embedding(vocab_size, hidden_size)
    nn.init.xavier_uniform_(self.in_emb.weight)
    nn.init.xavier_uniform_(self.out_emb.weight)

  #def forward(self, contexts, t):
  #  con_emb = self.in_emb(contexts).sum(dim=1, keepdim=True)
  #  target_emb = self.out_emb(t).transpose(2,1)
  #  print(con_emb.shape, target_emb.shape)
  #  y = con_emb.matmul(target_emb)
  #  return y
  def forward(self, contexts, t):
      con_emb = self.in_emb(contexts).sum(dim=1)          # [B, H]
      target_emb = self.out_emb(t)                         # [B, T, H]
      y = torch.bmm(con_emb.unsqueeze(1),                 # [B, 1, H]
                    target_emb.transpose(1, 2))            # [B, H, T]
      return y.squeeze(1)                                  # [B, T]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

vocab_size = train_dataset.vocab_size()
hidden_size = 3
model = CbowModel(vocab_size, hidden_size).to(device)

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 设置损失函数
# 集成了sigmoid的二元交叉熵误差函数
#loss_fn = nn.BCEWithLogitsLoss()
loss_fn = nn.CrossEntropyLoss()

# 训练
max_epoch = 10
for epoch in range(max_epoch) :
  for inputs, targets, labels in train_loader :
    inputs = inputs.to(device)
    targets = targets.to(device)
    labels = labels.to(device).squeeze(1)

    y = model.forward(inputs, targets)
    loss = loss_fn(y, labels)

    loss.backward()
    # deone中的update
    optimizer.step()
    optimizer.zero_grad()
    print("epoch:{}, loss:{}".format(epoch, loss))
