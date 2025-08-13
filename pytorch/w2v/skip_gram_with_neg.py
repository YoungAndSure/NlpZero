#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from common.util import *
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataset.easy_data import HelloW2vWithNegDataset
from torch.utils.data import DataLoader
from torchviz import make_dot

open_graph = False

device = "cuda" if torch.cuda.is_available() else "cpu"


train_dataset = HelloW2vWithNegDataset()
test_dataset = HelloW2vWithNegDataset()

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# 初始化模型
class SkipGramModel(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.in_emb = nn.Embedding(vocab_size, hidden_size)
    self.out_emb = nn.Embedding(vocab_size, hidden_size)

  def forward(self, center, target, label):
    BATCH = center.shape
    center_emb = self.in_emb(center)
    BATCH,HIDDEN_SIZE = center_emb.shape

    target_emb = self.out_emb(target)

    y = torch.sum(center_emb * target_emb, dim=1)
    y = y * label

    return y

  def predict(self, center) :
    BATCH = center.shape
    center_emb = self.in_emb(center)
    BATCH,HIDDEN_SIZE = center_emb.shape
    targets = torch.arange(0, self.vocab_size).to(device)
    target_embs = self.out_emb(targets)
    y = torch.matmul(center_emb, target_embs.T)
    values, indices = torch.topk(y, k=2, dim=1)
    return indices

vocab_size = train_dataset.vocab_size()
hidden_size = 10
model = SkipGramModel(vocab_size, hidden_size).to(device)

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# 损失函数
loss_fn = nn.LogSigmoid()

first_run=True
# 训练
max_epoch = 1000 # 负采样近似导致收敛速度变慢，需要加大训练次数
for epoch in range(max_epoch) :
  total_sample = 0
  total_loss = 0
  for context, center, label in train_loader :
    #print(context, center, label)
    context = context.to(device)
    center = center.to(device)
    label = label.to(device)

    batch_logit = model.forward(center, context, label)
    loss = loss_fn(batch_logit)
    loss = -loss.sum()

    if (open_graph and first_run) :
      vis_graph = make_dot(loss, params=dict(model.named_parameters()))
      vis_graph.render('.cbow_v1_model', format='png')  # 输出图像文件
      first_run=False

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    total_loss += loss.data
    total_sample += center.shape[0]

  print("epoch:{}, avg_loss:{}".format(epoch, total_loss/total_sample))

with torch.no_grad() :
  print("text:{}".format(train_dataset.get_text()))
  centers = ['goodbye','i']
  for center in centers :
    id = torch.tensor(train_dataset.to_ids(center))[torch.newaxis].to(device)
    contexts_id = model.predict(id)
    contexts_word = []
    for i in range(contexts_id.shape[0]) :
      for j in range(contexts_id.shape[1]) :
        contexts_word.append(train_dataset.to_word(contexts_id[i][j].item()))
    print(center, contexts_word)