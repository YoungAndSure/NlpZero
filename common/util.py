#! python3

import numpy as np

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

      if left_i > 0 :
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