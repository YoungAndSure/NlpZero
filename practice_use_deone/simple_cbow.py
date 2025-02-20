#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.util import *
import matplotlib.pyplot as plt
from deone import *

text = 'You say goodbye and I say hello.'

# 分词，转id
corpus, word2id, id2word = preprocess(text)

# 生成上下文
ori_contexts, ori_targets = create_contexts_target(corpus=corpus, window_size=1)
for i in range(len(ori_targets)) :
  print("target:{}, contexts:{},{}".format(id2word[ori_targets[i]], id2word[ori_contexts[i][0]], id2word[ori_contexts[i][1]]))

# one-hot编码
vocab_size = len(word2id)
contexts = convert_one_hot(ori_contexts, vocab_size)
targets = convert_one_hot(ori_targets, vocab_size)

# 初始化模型
hidden_size = 3
model = SimpleCbow(hidden_size, vocab_size)

# 设置优化器
optimizer = Momentum(lr=0.1)
optimizer.setup(model)

# 训练
max_epoch = 100
for epoch in range(max_epoch) :
  for i in range(len(targets)) :
    x = Parameter(np.array(contexts[i])[:, np.newaxis])
    t = Parameter(np.array(targets[i])[:, np.newaxis])
    y = model.forward(x)
    # 多分类问题是说，有这么多分类，我想知道哪个概率是最高的，
    # 所以它需要计算所有分类的概率，然后拿出最高概率的，和label计算loss
    # 所以用的是softmax，某一个分类的概率要和所有其他分类的概率联系起来
    loss = softmax_cross_entropy_simple(y, t)
    loss.backward()
    optimizer.update()
    print("epoch:{}, loss:{}".format(epoch, loss.data))
