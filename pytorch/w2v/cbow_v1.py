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
    self.out_emb = nn.Embedding(vocab_size, hidden_size)

  def forward(self, contexts, center):
    BATCH,WINDOW_SIZE = contexts.shape
    BATCH = center.shape
    context_emb = self.in_emb(contexts)
    BATCH,WINDOW_SIZE,HIDDEN_SIZE = context_emb.shape
    center_emb = self.out_emb(center[:,torch.newaxis])
    BATCH,_,HIDDEN_SIZE = center_emb.shape

    context_mean = torch.mean(context_emb, dim=1, keepdim=True)
    BATCH,ONE,HIDDEN_SIZE = context_mean.shape
    center_context_mul = torch.matmul(center_emb, context_mean.transpose(1,2)).squeeze()
    BATCH = center_context_mul.shape

    vocab_words_emb = self.out_emb(torch.arange(self.vocab_size).to(device))
    VOCAB_SIZE,HIDDEN_SIZE = vocab_words_emb.shape
    vocab_context_mul =  torch.matmul(context_mean, vocab_words_emb.t())
    vocab_context_logsumexp = torch.logsumexp(vocab_context_mul, dim=2, keepdim=True).squeeze()

    loss = -(center_context_mul - vocab_context_logsumexp)
    return loss

  def predict(self, contexts) :
    BATCH,WINDOW_SIZE = contexts.shape
    context_emb = self.in_emb(contexts)
    BATCH,WINDOW_SIZE,HIDDEN_SIZE = context_emb.shape

    context_mean = torch.mean(context_emb, dim=1, keepdim=True)
    BATCH,ONE,HIDDEN_SIZE = context_mean.shape

    vocab_words_emb = self.out_emb(torch.arange(self.vocab_size).to(device))
    VOCAB_SIZE,HIDDEN_SIZE = vocab_words_emb.shape
    vocab_context_mul =  torch.matmul(context_mean, vocab_words_emb.t())
    y = torch.argmax(vocab_context_mul.squeeze(1), dim=1, keepdim=True)
    return y

device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_size = train_dataset.vocab_size()
hidden_size = 10
model = CbowModel(vocab_size, hidden_size).to(device)

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

first_run=True
# 训练
max_epoch = 100
for epoch in range(max_epoch) :
  total_sample = 0
  total_loss = 0
  for contexts, center in train_loader :
    contexts = contexts.to(device)
    center = center.to(device)

    batch_loss = model.forward(contexts, center)
    loss = batch_loss.sum()

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

contexts = [['i','hello'],['i','goodbye']]
for context in contexts :
  ids = torch.tensor(train_dataset.to_ids(context))[torch.newaxis].to(device)
  center_id = model.predict(ids)
  for i in range(center_id.shape[0]) :
    center_word = train_dataset.to_word(center_id[i][0].item())
  print(context, center_word)