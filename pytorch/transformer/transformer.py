#! python3

import torch
from torch import nn
import numpy as np

class PositionEncoder(nn.Module) :
  def __init__(self) :
    super().__init__()

  def forward(self, xs) :
    BATCH, SEQ_LEN, WORDVEC_SIZE = xs.shape[0], xs.shape[1], xs.shape[2]

    P = np.zeros((SEQ_LEN, WORDVEC_SIZE))
    for pos in range(SEQ_LEN) :
      for i in range(WORDVEC_SIZE // 2) :
        P[pos][2 * i] = np.sin(pos / np.power(10000, (2 * i) / WORDVEC_SIZE))
        P[pos][2 * i + 1] = np.cos(pos / np.power(10000, (2 * i) / WORDVEC_SIZE))
      if WORDVEC_SIZE % 2 != 0 :
        P[pos][WORDVEC_SIZE - 1] = np.sin(pos / np.power(10000, (WORDVEC_SIZE - 1) / WORDVEC_SIZE))
    P = torch.tensor(np.expand_dims(P, 0))

    assert((xs.shape[1], xs.shape[2]) == (P.shape[1], P.shape[2]))
    # 自动对batch广播
    y = xs + P
    return y

class TransformerEncoder(nn.Module) :
  def __init__(self, vocab_size, seq_len, wordvec_size) :
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, wordvec_size)
    self.pe = PositionEncoder()

  def forward(self, xs) :
    embs = self.embedding(xs)
    embs_with_pe = self.pe(embs)

    return embs_with_pe

vocab_size = 100
seq_len = 3
wordvec_size = 9
batch_size = 2
xs = torch.randint(0, vocab_size, (batch_size, seq_len))

encoder = TransformerEncoder(vocab_size, seq_len, wordvec_size)
encode_output = encoder(xs)
print(encode_output.shape)