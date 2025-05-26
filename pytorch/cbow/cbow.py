#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from common.util import *
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataset.easy_data import EasyDataset
from torch.utils.data import DataLoader
from torchviz import make_dot

open_graph = False

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

  def forward(self, contexts, t):
    con_emb = self.in_emb(contexts).sum(dim=1, keepdim=True)
    target_emb = self.out_emb(t).transpose(2,1)
    y = con_emb.matmul(target_emb)
    return y

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

vocab_size = train_dataset.vocab_size()
hidden_size = 10
model = CbowModel(vocab_size, hidden_size).to(device)

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 设置损失函数
# 集成了sigmoid的二元交叉熵误差函数
loss_fn = nn.BCEWithLogitsLoss()

first_run=True
# 训练
max_epoch = 100
for epoch in range(max_epoch) :
  for inputs, targets, labels in train_loader :
    inputs = inputs.to(device)
    targets = targets.to(device)
    labels = labels.to(device)

    y = model.forward(inputs, targets)
    loss = loss_fn(y, labels)

    if (open_graph and first_run) :
      vis_graph = make_dot(y, params=dict(model.named_parameters()))
      vis_graph.render('cbow_model', format='png')  # 输出图像文件
      first_run=False

    loss.backward()
    # deone中的update
    optimizer.step()
    optimizer.zero_grad()
    print("epoch:{}, loss:{}".format(epoch, loss))
