#! python3

import torch
from torch import nn
import numpy as np
import math
from transformer import Transformer

vocab_size = 100
seq_len = 3
d_model = 64
dim_feedforward = d_model * 4
nhead = 8
batch_size = 2
encoder_layer = 6
decoder_layer = 6
xs = torch.randint(0, vocab_size, (batch_size, seq_len))
ts = torch.randint(0, vocab_size, (batch_size, seq_len))
sos = torch.randint(0, vocab_size, (batch_size, 1))
decode_input = torch.concat((sos, ts), dim=1)
eos = torch.randint(0, vocab_size, (batch_size, 1))
label = torch.concat((eos, ts), dim=1)

transformer = Transformer(vocab_size, d_model, nhead, dim_feedforward, encoder_layer, decoder_layer)

ys = transformer(xs, decode_input)
print(ys.shape, label.shape)
