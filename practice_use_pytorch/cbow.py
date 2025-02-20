#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.util import *
import matplotlib.pyplot as plt
import torch
from torch import nn

text = 'You say goodbye and I say hello.'

# 分词，转id
corpus, word2id, id2word = preprocess(text)

# 生成上下文
ori_contexts, ori_targets = create_contexts_target(corpus=corpus, window_size=1)
'''
for i in range(len(ori_targets)) :
  print("target:{}, contexts:{},{}".format(id2word[ori_targets[i]], id2word[ori_contexts[i][0]], id2word[ori_contexts[i][1]]))
'''

# 初始化模型
class CbowModel(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super().__init__()
    self.in_emb = nn.Embedding(vocab_size, hidden_size)
    self.out_emb = nn.Embedding(vocab_size, hidden_size)
    self.activator = nn.Sigmoid()

  def forward(self, contexts, t):
    con_emb = self.in_emb(contexts).sum(dim=1)
    target_emb = self.out_emb(t).squeeze(1).T
    y = con_emb.matmul(target_emb)
    return y

vocab_size = len(word2id)
hidden_size = 3
model = CbowModel(vocab_size, hidden_size)

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 初始化负样本采样
unigram_sampler = UnigramSampler(corpus, 0.75, 5)

# 设置损失函数
# 集成了sigmoid的二元交叉熵误差函数
loss_fn = nn.BCEWithLogitsLoss()

# 训练
max_epoch = 10
for epoch in range(max_epoch) :
  for i in range(len(ori_targets)) :
    x = torch.from_numpy(np.array([ori_contexts[i]]))
    pt = torch.from_numpy(np.array([[ori_targets[i]]]))
    y = model.forward(x, pt)
    loss = loss_fn(y, torch.ones((1, 1)))

    negatives = unigram_sampler.get_negative_sample(np.array([ori_targets[i]]))[0]
    for negative in negatives :
      nt = torch.from_numpy(np.array([negative], dtype=np.long))
      y = model.forward(x, nt)
      loss += loss_fn(y, torch.zeros((1, 1)))

    loss.backward()
    # deone中的update
    optimizer.step()
    optimizer.zero_grad()
    print("epoch:{}, loss:{}".format(epoch, loss))
