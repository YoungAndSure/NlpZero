#! python3

import torch
from torch import nn
import numpy as np
import math

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

class SingleHeadAttention(nn.Module) :
  def __init__(self, embed_dim) :
    super().__init__()
    self.WQ = nn.LazyLinear(embed_dim, dtype=torch.float64)
    self.WK = nn.LazyLinear(embed_dim, dtype=torch.float64)
    self.WV = nn.LazyLinear(embed_dim, dtype=torch.float64)
    self.sqrt_embed_dim = math.sqrt(embed_dim)

  def forward(self, xs) :
    BATCH, SEQ_LEN, HIDDEN = xs.shape[0], xs.shape[1], xs.shape[2]

    k = self.WQ(xs)
    q = self.WQ(xs)
    v = self.WV(xs)

    qk = torch.matmul(q, k.transpose(1,2))
    qk = qk / self.sqrt_embed_dim

    qk_softmax = torch.softmax(qk, dim=2)

    BATCH, SEQ_LEN, SEQ_LEN = qk_softmax.shape[0], qk_softmax.shape[1], qk_softmax.shape[2]
    BATCH, SEQ_LEN, EMBED_DIM = v.shape[0], v.shape[1], v.shape[2]

    y = torch.matmul(qk_softmax, v)

    return y

class MultiHeadAttention(nn.Module) :
  def __init__(self, embed_dim, head_num) :
    super().__init__()
    assert(embed_dim // head_num * head_num == embed_dim)
    self.multi_head_attention = nn.ModuleList([SingleHeadAttention(embed_dim//head_num) for _ in range(head_num)])

  def forward(self, xs) :
    ys = None
    for single_head_attention in self.multi_head_attention :
      if ys is None :
        ys = single_head_attention(xs)
      else :
        ys = torch.concat((ys, single_head_attention(xs)), dim=2)
    BATCH_SIZE, SEQ_LEN, EMBED_DIM = ys.shape
    return ys

class TransformerEncoderLayer(nn.Module) :
  def __init__(self) :
    super().__init__()
    self.multi_head_attention = MultiHeadAttention(embed_dim=64, head_num=8)
  
  def forward(self, xs) :
    ys = self.multi_head_attention(xs)
    return ys

class TransformerEncoder(nn.Module) :
  def __init__(self, encoder_layer=1) :
    super().__init__()
    self.encoder_layer = nn.ModuleList([TransformerEncoderLayer() for _ in range(encoder_layer)])

  def forward(self, xs) :
    BATCH_SIZE, SEQ_LEN, WORDVEC_SIZE = xs.shape
    for el in self.encoder_layer :
      ys = el(xs)
      xs = ys
    return ys

class Transformer(nn.Module) :
  def __init__(self, vocab_size, wordvec_size) :
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, wordvec_size)
    self.pe = PositionEncoder()
    self.encoder = TransformerEncoder()

  def forward(self, xs) :
    embs = self.embedding(xs)
    embs_with_pe = self.pe(embs)
    encode = self.encoder(embs_with_pe)

    return encode

vocab_size = 100
seq_len = 3
wordvec_size = 9
batch_size = 2
xs = torch.randint(0, vocab_size, (batch_size, seq_len))

transformer = Transformer(vocab_size, wordvec_size)
ys = transformer(xs)
print(ys.shape)