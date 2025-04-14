#! python3

import torch
from torch import nn
import numpy as np
import math

class PositionEncoder(nn.Module) :
  def __init__(self) :
    super().__init__()

  def forward(self, xs) :
    BATCH, SEQ_LEN, D_MODEL = xs.shape[0], xs.shape[1], xs.shape[2]

    P = np.zeros((SEQ_LEN, D_MODEL))
    for pos in range(SEQ_LEN) :
      for i in range(D_MODEL // 2) :
        P[pos][2 * i] = np.sin(pos / np.power(10000, (2 * i) / D_MODEL))
        P[pos][2 * i + 1] = np.cos(pos / np.power(10000, (2 * i) / D_MODEL))
      if D_MODEL % 2 != 0 :
        P[pos][D_MODEL - 1] = np.sin(pos / np.power(10000, (D_MODEL - 1) / D_MODEL))
    # TODO: 改用tensor的方法实现，不用单独处理
    P = torch.tensor(np.expand_dims(P, 0)).to('cuda')

    assert((xs.shape[1], xs.shape[2]) == (P.shape[1], P.shape[2]))
    # 自动对batch广播
    y = xs + P
    return y

class SingleHeadAttention(nn.Module) :
  def __init__(self, embed_dim, mask=False) :
    super().__init__()
    self.WQ = nn.LazyLinear(embed_dim, dtype=torch.float64)
    self.WK = nn.LazyLinear(embed_dim, dtype=torch.float64)
    self.WV = nn.LazyLinear(embed_dim, dtype=torch.float64)
    self.sqrt_embed_dim = math.sqrt(embed_dim)
    self.mask = mask

  def forward(self, xs, encode=None) :
    BATCH, SEQ_LEN, HIDDEN = xs.shape[0], xs.shape[1], xs.shape[2]

    q = self.WQ(xs)
    k = self.WK(xs) if encode is None else self.WK(encode)
    v = self.WV(xs) if encode is None else self.WV(encode)

    qk = torch.matmul(q, k.transpose(1,2))
    qk = qk / self.sqrt_embed_dim
    
    if self.mask :
      assert(qk.shape[1] == qk.shape[2])
      mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
      qk[mask.broadcast_to(BATCH, mask.shape[0], mask.shape[1])] = -torch.inf

    qk_softmax = torch.softmax(qk, dim=2)

    BATCH, SEQ_LEN, SEQ_LEN = qk_softmax.shape[0], qk_softmax.shape[1], qk_softmax.shape[2]
    BATCH, SEQ_LEN, EMBED_DIM = v.shape[0], v.shape[1], v.shape[2]

    y = torch.matmul(qk_softmax, v)

    return y

class MultiHeadAttention(nn.Module) :
  def __init__(self, d_model, nhead, mask=False) :
    super().__init__()
    assert(d_model // nhead * nhead == d_model)
    self.multi_head_attention = nn.ModuleList([SingleHeadAttention(d_model//nhead, mask) for _ in range(nhead)])

  def forward(self, xs, encode=None) :
    ys = None
    for single_head_attention in self.multi_head_attention :
      if ys is None :
        ys = single_head_attention(xs, encode)
      else :
        ys = torch.concat((ys, single_head_attention(xs)), dim=2)
    BATCH_SIZE, SEQ_LEN, EMBED_DIM = ys.shape
    return ys

class AddNorm(nn.Module) :
  def __init__(self, d_model) :
    super().__init__()
    self.layer_norm = nn.LayerNorm(d_model, dtype=torch.float64)

  def forward(self, xs, ys) :
    ys = ys + xs
    ys = self.layer_norm(ys)
    return ys

class FeedForwardNetwork(nn.Module) :
  def __init__(self, d_model, dim_feedforward) :
    super().__init__()
    self.linear1 = nn.LazyLinear(dim_feedforward, dtype=torch.float64)
    self.activate1 = nn.ReLU()
    self.linear2 = nn.LazyLinear(d_model, dtype=torch.float64)
    self.activate2 = nn.ReLU()

  def forward(self, xs) :
    BATCH, SEQ_LEN, D_MODEL = xs.shape
    ys_linear1 = self.linear1(xs)
    BATCH, SEQ_LEN, DIM_FEEDFORWARD = ys_linear1.shape
    ys_activate1 = self.activate1(ys_linear1)

    ys_linear2 = self.linear2(ys_activate1)
    ys_activate2 = self.activate2(ys_linear2)
    return ys_activate2

class TransformerEncoderLayer(nn.Module) :
  def __init__(self, d_model, nhead, dim_feedforward) :
    super().__init__()
    self.multi_head_attention = MultiHeadAttention(d_model, nhead, False)
    self.add_norm = AddNorm(d_model)
    # 先升维，再降维，不然没法做第二次残差了
    self.ffn = FeedForwardNetwork(d_model, dim_feedforward)
  
  def forward(self, xs) :
    BATCH, SEQ_LEN, D_MODEL = xs.shape
    ys_multi_head = self.multi_head_attention(xs)
    BATCH, SEQ_LEN, D_MODEL = ys_multi_head.shape
    assert(xs.shape == ys_multi_head.shape)

    ys_add_norm1 = self.add_norm(xs, ys_multi_head)

    ys_ffn = self.ffn(ys_add_norm1)

    ys_add_norm2 = self.add_norm(ys_add_norm1, ys_ffn)

    return ys_add_norm2

class TransformerEncoder(nn.Module) :
  def __init__(self, d_model, nhead, dim_feedforward, encoder_layer) :
    super().__init__()
    self.encoder_layer = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward) for _ in range(encoder_layer)])

  def forward(self, xs) :
    BATCH_SIZE, SEQ_LEN, D_MODEl = xs.shape
    for el in self.encoder_layer :
      ys = el(xs)
      xs = ys
    return ys

class TransformerDecoderLayer(nn.Module) :
  def __init__(self, d_model, nhead, dim_feedforward) :
    super().__init__()
    self.mask_multi_head_attention = MultiHeadAttention(d_model, nhead, True)
    self.add_norm1 = AddNorm(d_model)
    self.multi_head_attention = MultiHeadAttention(d_model, nhead, False)
    self.add_norm2 = AddNorm(d_model)
    self.ffn = FeedForwardNetwork(d_model, dim_feedforward)
    self.add_norm3 = AddNorm(d_model)

  def forward(self, encode, xs) :
    ys_mask_multi_head = self.mask_multi_head_attention(xs)
    ys_add_norm1 = self.add_norm1(xs, ys_mask_multi_head)
    ys_multi_head = self.multi_head_attention(ys_add_norm1, encode)
    ys_add_norm2 = self.add_norm2(ys_add_norm1, ys_multi_head)
    ys_ffn = self.ffn(ys_add_norm2)
    ys_add_norm3 = self.add_norm3(ys_add_norm2, ys_ffn)
    return ys_add_norm3

class TransformerDecoder(nn.Module) :
  def __init__(self, vocab_size, d_model, nhead, dim_feedforward, decoder_layer) :
    super().__init__()
    self.decoder_layer = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward) for _ in range(decoder_layer)])
    self.linear = nn.LazyLinear(vocab_size, dtype=torch.float64)

  def forward(self, encode, xs) :
    for dl in self.decoder_layer :
      ys = dl(encode, xs)
      xs = ys
    ys_linear = self.linear(ys)
    # NOTE:nn.CrossEntropyLoss输入要求是logit，不能在这softmax
    #ys_softmax = torch.softmax(ys_linear, dim=2)
    #ys_argmax = torch.argmax(ys_softmax, dim=2)
    return ys_linear

class Transformer(nn.Module) :
  def __init__(self, vocab_size, d_model, nhead, dim_feedforward, encoder_layer, decoder_layer) :
    super().__init__()
    self.encode_embedding = nn.Embedding(vocab_size, d_model)
    self.decode_embedding = nn.Embedding(vocab_size, d_model)
    self.pe = PositionEncoder()
    self.encoder = TransformerEncoder(d_model, nhead, dim_feedforward, encoder_layer)
    self.decoder = TransformerDecoder(vocab_size, d_model, nhead, dim_feedforward, decoder_layer)

  def forward(self, xs, ts) :
    encode_embs = self.encode_embedding(xs)
    encode_embs_with_pe = self.pe(encode_embs)
    encode = self.encoder(encode_embs_with_pe)

    decode_embs = self.decode_embedding(ts)
    decode_embs_with_pe = self.pe(decode_embs)
    decode = self.decoder(encode, decode_embs_with_pe)

    return decode