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

    P = torch.zeros((SEQ_LEN, D_MODEL), dtype=torch.float64, device=xs.device)
    for pos in range(SEQ_LEN) :
      for i in range(D_MODEL // 2) :
        P[pos][2 * i] = torch.sin(torch.tensor(pos / (10000 ** (2 * i) / D_MODEL), device=xs.device))
        P[pos][2 * i + 1] = torch.cos(torch.tensor(pos / (10000 ** (2 * i) / D_MODEL), device=xs.device))
      if D_MODEL % 2 != 0 :
        P[pos][D_MODEL - 1] = torch.sin(torch.tensor(pos / (10000 ** (D_MODEL - 1) / D_MODEL), device=xs.device))

    assert((xs.shape[1], xs.shape[2]) == (P.shape[0], P.shape[1]))
    # 自动对batch广播
    y = xs + P.unsqueeze(0).broadcast_to(BATCH, SEQ_LEN, D_MODEL)
    return y

class MultiHeadAttention(nn.Module) :
  def __init__(self, d_model, nhead, mask=False) :
    super().__init__()
    assert(d_model // nhead * nhead == d_model)
    self.nhead = nhead
    self.d_k = d_model // nhead
    self.WQ = nn.Linear(d_model, d_model, dtype=torch.float64, bias=False)
    self.WK = nn.Linear(d_model, d_model, dtype=torch.float64, bias=False)
    self.WV = nn.Linear(d_model, d_model, dtype=torch.float64, bias=False)
    self.fc = nn.Linear(d_model, d_model, dtype=torch.float64)
    self.sqrt_embed_dim = math.sqrt(self.d_k)
    self.mask = mask

  def forward(self, Q, K, V) :
    # qkv.shape 在encoder中相同，在decoder中，kv.shape相同，q.shape和kv.shape不同
    assert(K.shape == V.shape), "{}, {}".format(K.shape, V.shape)
    BATCH, Q_SEQ_LEN, D_MODEL = Q.shape
    BATCH, KV_SEQ_LEN, D_MODEL = K.shape

    q = self.WQ(Q)
    k = self.WK(K)
    v = self.WV(V)

    multi_head_q = q.view(BATCH, Q_SEQ_LEN, self.nhead, self.d_k).transpose(1, 2) # BATCH, nhead, Q_SEQ_LEN, d_k
    multi_head_k = k.view(BATCH, KV_SEQ_LEN, self.nhead, self.d_k).transpose(1, 2) # BATCH, nhead, KV_SEQ_LEN, d_k
    multi_head_v = v.view(BATCH, KV_SEQ_LEN, self.nhead, self.d_k).transpose(1, 2)

    multi_head_qk = torch.matmul(multi_head_q, multi_head_k.transpose(2, 3))
    multi_head_qk = multi_head_qk / self.sqrt_embed_dim

    BATCH, N_HEAD, Q_SEQ_LEN, KV_SEQ_LEN = multi_head_qk.shape

    if self.mask :
      mask = torch.triu(torch.ones(Q_SEQ_LEN, KV_SEQ_LEN, device=multi_head_qk.device), diagonal=1).bool()
      multi_head_qk[mask.broadcast_to(BATCH, self.nhead, mask.shape[0], mask.shape[1])] = -torch.inf

    multi_head_qk_softmax = torch.softmax(multi_head_qk, dim=-1)

    BATCH, N_HEAD, Q_SEQ_LEN, KV_SEQ_LEN = multi_head_qk_softmax.shape
    BATCH, N_HEAD, KV_SEQ_LEN, EMBED_DIM = multi_head_v.shape

    y = torch.matmul(multi_head_qk_softmax, multi_head_v)
    BATCH, N_HEAD, Q_SEQ_LEN, EMBED_DIM = y.shape

    y = y.transpose(1, 2)
    BATCH, Q_SEQ_LEN, N_HEAD, EMBED_DIM = y.shape
    y = y.reshape(BATCH, Q_SEQ_LEN, D_MODEL)
    y = self.fc(y)

    return y

class AddNorm(nn.Module) :
  def __init__(self, d_model) :
    super().__init__()
    self.layer_norm = nn.LayerNorm(d_model, dtype=torch.float64)

  def forward(self, xs, ys) :
    ys = ys + xs
    ys = self.layer_norm(ys)
    return ys

class FeedForwardNetwork(nn.Module) :
  # formula: FFN(x) = max(0, x * W1 + b1) * W2 + b2
  def __init__(self, d_model, dim_feedforward) :
    super().__init__()
    self.linear1 = nn.LazyLinear(dim_feedforward, dtype=torch.float64)
    self.activate1 = nn.ReLU()
    self.linear2 = nn.LazyLinear(d_model, dtype=torch.float64)

  def forward(self, xs) :
    BATCH, SEQ_LEN, D_MODEL = xs.shape
    ys_linear1 = self.linear1(xs)
    BATCH, SEQ_LEN, DIM_FEEDFORWARD = ys_linear1.shape
    ys_activate1 = self.activate1(ys_linear1)

    ys_linear2 = self.linear2(ys_activate1)
    return ys_linear2

class TransformerEncoderLayer(nn.Module) :
  def __init__(self, d_model, nhead, dim_feedforward) :
    super().__init__()
    self.multi_head_attention = MultiHeadAttention(d_model, nhead, False)
    self.add_norm = AddNorm(d_model)
    # 先升维，再降维，不然没法做第二次残差了
    self.ffn = FeedForwardNetwork(d_model, dim_feedforward)
  
  def forward(self, xs) :
    BATCH, SEQ_LEN, D_MODEL = xs.shape
    ys_multi_head = self.multi_head_attention(xs, xs, xs)
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
    ys_mask_multi_head = self.mask_multi_head_attention(xs, xs, xs)
    ys_add_norm1 = self.add_norm1(xs, ys_mask_multi_head)
    ys_multi_head = self.multi_head_attention(ys_add_norm1, encode, encode) # Q=ys_add_norm1, K=encode, V=encode
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

  def generate(self, xs, startid, max_seq_len, endid=None) :
    encode_embs = self.encode_embedding(xs)
    encode_embs_with_pe = self.pe(encode_embs)
    encode = self.encoder(encode_embs_with_pe)

    ts = torch.tensor(startid).broadcast_to(xs.shape[0], 1).to('cuda')
    while ts.shape[1] < max_seq_len:
      decode_embs = self.decode_embedding(ts)
      decode_embs_with_pe = self.pe(decode_embs)
      decode = self.decoder(encode, decode_embs_with_pe)

      ys_softmax = torch.softmax(decode, dim=2)
      ys_argmax = torch.argmax(ys_softmax, dim=2)

      last_word = ys_argmax[:, -1].unsqueeze(-1)
      ts = torch.concat((ts, last_word), dim=1)

    return ts