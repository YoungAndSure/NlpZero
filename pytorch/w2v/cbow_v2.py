#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from common.util import *
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataset.easy_data import HelloW2vDataset
from torch.utils.data import DataLoader
from torchviz import make_dot

open_graph = False

device = "cuda" if torch.cuda.is_available() else "cpu"


train_dataset = HelloW2vDataset()
test_dataset = HelloW2vDataset()

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# 初始化模型
class CbowModel(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.in_emb = nn.Embedding(vocab_size, hidden_size)
    self.out_emb = nn.Linear(hidden_size, vocab_size)

  def forward(self, contexts):
    BATCH,WINDOW_SIZE = contexts.shape
    context_emb = self.in_emb(contexts)
    BATCH,WINDOW_SIZE,HIDDEN_SIZE = context_emb.shape
    context_avg_emb = torch.mean(context_emb, dim=1)

    y = self.out_emb(context_avg_emb)

    return y

  def predict(self, contexts) :
    logit = self.forward(contexts)
    y = torch.argmax(logit.squeeze(1), dim=1, keepdim=True)
    return y

vocab_size = train_dataset.vocab_size()
hidden_size = 10
model = CbowModel(vocab_size, hidden_size).to(device)

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# 损失函数
loss_fn = nn.CrossEntropyLoss()

first_run=True
# 训练
max_epoch = 100
for epoch in range(max_epoch) :
  total_sample = 0
  total_loss = 0
  for contexts, target in train_loader :
    contexts = contexts.to(device)
    target = target.to(device)

    batch_logit = model.forward(contexts)
    loss = loss_fn(batch_logit, target)

    if (open_graph and first_run) :
      vis_graph = make_dot(loss, params=dict(model.named_parameters()))
      vis_graph.render('.cbow_v1_model', format='png')  # 输出图像文件
      first_run=False

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    total_loss += loss.data
    total_sample += target.shape[0]

  print("epoch:{}, avg_loss:{}".format(epoch, total_loss/total_sample))

with torch.no_grad() :
  contexts = [['i','hello'],['i','goodbye']]
  for context in contexts :
    ids = torch.tensor(train_dataset.to_ids(context))[torch.newaxis].to(device)
    center_id = model.predict(ids)
    for i in range(center_id.shape[0]) :
      center_word = train_dataset.to_word(center_id[i][0].item())
    print(context, center_word)