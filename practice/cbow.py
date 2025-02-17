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

# one-hot编码, 不用了
vocab_size = len(word2id)
#contexts = convert_one_hot(ori_contexts, vocab_size)
#targets = convert_one_hot(ori_targets, vocab_size)

# 初始化模型
hidden_size = 3
model = Cbow(hidden_size, vocab_size)

# 设置优化器
optimizer = Momentum(lr=0.1)
optimizer.setup(model)

# 初始化负样本采样
unigram_sampler = UnigramSampler(corpus, 0.75, 5)

# 训练
max_epoch = 100
for epoch in range(max_epoch) :
  for i in range(len(ori_targets)) :
    x = Parameter(np.array(ori_contexts[i])[:, np.newaxis])
    positive = Parameter(np.array([ori_targets[i]]))
    y = model.forward(x, positive)
    loss = sigmoid_cross_entropy_simple(y, np.ones(1, np.int32))

    negatives = unigram_sampler.get_negative_sample(np.array([ori_targets[i]]))
    for negative in negatives :
      y = model.forward(x, negative)
      loss += sigmoid_cross_entropy_simple(y, np.zeros(1, np.int32))

    loss.backward()
    optimizer.update()
    print("epoch:{}, loss:{}".format(epoch, loss.data))
