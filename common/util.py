#! python3

import numpy as np
import collections
import time
from torch import nn
import os
import torch

def preprocess(text) :
  text = text.lower()
  text = text.replace('.', ' .')
  words = text.split(' ')

  word2id={}
  id2word={}
  for word in words :
    if word not in word2id :
      new_id = len(word2id)
      word2id[word] = new_id
      id2word[new_id] = word

  corpus = [word2id[w] for w in words]

  return corpus, word2id, id2word

def create_co_matrix(corpus, vocab_size, window_size=1) :
  corpus_size = len(corpus)
  co_matrix = np.zeros((vocab_size, vocab_size)).astype(np.float32)

  for index,id in enumerate(corpus) :
    for i in range(1, window_size + 1) :
      left_i = index - i
      right_i = index + i

      if left_i >= 0 :
        left_word_id = corpus[left_i]
        co_matrix[id][left_word_id] += 1
      if right_i < corpus_size :
        right_word_id = corpus[right_i]
        co_matrix[id][right_word_id] += 1
  
  return co_matrix

def cos_similarity(x, y, eps=1e-8) :
  x = x / np.sqrt(np.sum(x ** 2) + 1e-8)
  y = y / np.sqrt(np.sum(y ** 2) + 1e-8)
  return np.dot(x, y)

def most_similar(query, word2id, id2word, word_matrix, top=5) :
  query = query.lower()
  if query not in word2id :
    print("no query in dict:{}".format(query))
    return
  
  query_id = word2id[query]
  query_vector = word_matrix[query_id]

  vocab_size = len(id2word)
  similarity = np.zeros(vocab_size)
  for i in range(vocab_size) :
    similarity[i] = cos_similarity(query_vector, word_matrix[i])

  count = 0
  for i in (-1 * similarity).argsort() :
    if id2word[i] == query :
      continue
    print("word:{}, id:{}, vector:{}, similar:{}".format(id2word[i], i, word_matrix[i], similarity[i]))
    count += 1
    if count > top :
      return

def ppmi(C, verbose=True, eps=1e-8) :
  M = np.zeros_like(C, dtype=np.float32)
  S = np.sum(C, axis=0)
  N = np.sum(C)
  total = C.shape[0] * C.shape[1]
  cnt = 0

  for i in range(C.shape[0]) :
    for j in range(C.shape[1]) :
      M[i][j] = np.log2((C[i][j] * N) / (S[i] * S[j]) + eps)
      M[i][j] = max(M[i][j], 0)
      
      if verbose :
        cnt += 1
        if cnt % 100 == 0:
          print("processing:{}%".format(cnt / total * 100))
  
  return M

def create_contexts_target(corpus, window_size=1, return_numpy=False) :
  targets = [[target] for target in corpus[window_size:-window_size]]
  contexts = []

  for idx in range(window_size, len(corpus) - window_size) :
    cs = []
    for t in range(-window_size, window_size + 1) :
      if t == 0 :
        continue
      cs.append(corpus[idx + t])
    contexts.append(cs)
  if return_numpy :
    return np.array(contexts), np.array(targets)
  return contexts, targets

def convert_one_hot(input, vec_size) :
  input = np.array(input)
  input_shape = input.shape
  output_shape = [i for i in input_shape]
  output_shape.append(vec_size)
  input = input.reshape(-1)
  output = []
  for num in input :
    o = [0 if i != num else 1 for i in range(0, vec_size)]
    output.append(o)
  return np.array(output).reshape(output_shape).tolist()

class UnigramSampler :
  def __init__(self, corpus, power, sample_size) :
    self.sample_size = sample_size
    counter = collections.Counter(corpus)
    self.vocab_size = len(counter)
    self.word_p = np.zeros(self.vocab_size, dtype = np.int32)
    for i in range(self.vocab_size) :
      self.word_p[i] = counter[i]
    self.word_p = np.power(self.word_p, power)
    self.word_p /= np.sum(self.word_p)

  def get_negative_sample(self, targets) :
    batch_size = targets.shape[0]
    results = np.zeros((batch_size, self.sample_size), dtype=np.int32)
    for i in range(batch_size) :
      p = self.word_p.copy()
      target_idx = targets[i]
      p[target_idx] = 0
      p /= np.sum(p)
      results[i, :] = np.random.choice(self.vocab_size, self.sample_size, replace=False, p=p)
    return results

class CostRecorder :
  def __init__(self) :
    self.last_time = time.perf_counter()
    self.cost = []
    self.name = []

  def record(self, interval_name) :
    cur_time = time.perf_counter()
    self.cost.append(cur_time - self.last_time)
    self.last_time = cur_time

    self.name.append(interval_name)

  def print_record(self) :
    print()
    print("-----------cost-----------")
    for i in range(len(self.cost)) :
      print("{} : {:.3f} ms".format(self.name[i], self.cost[i] * 1000))
    print("total : {:.3f} ms".format(sum(self.cost) * 1000))
    print("-----------end-----------")

def save_model(model: nn.Module, save_path: str, force_override=True):
    if os.path.exists(save_path):
        if force_override:
            os.remove(save_path)
            print(f"deleted: {save_path}")
        else:
            raise FileExistsError(f"file exist: {save_path}")
    
    torch.save(model.state_dict(), save_path)
    print(f"model saved: {save_path}")