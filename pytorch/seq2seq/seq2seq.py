#! python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from common.util import CostRecorder, save_model
import torch
import numpy as np
from dataset.addition import AdditionDataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Embedding
from torch.nn import LSTM
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

# config:
retrain_and_dump=False
batch_size = 8
max_epoch = 2
file_name = "seq2seq.pth"
manual_test_case_size = 10

# start
recorder = CostRecorder()

train_data = AdditionDataset(data_type="train") 
test_data = AdditionDataset(data_type="test")
vocab_size = train_data.vocab_size()

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

print()
print("---------dataset message---------")
print("vocab_size:", vocab_size)
print("train data size:", len(train_dataloader) * batch_size)
print("test data size", len(test_dataloader) * batch_size)
for x,t in train_dataloader:
    print(f"Shape of train X [BATCH, H]: {x.shape}")
    print(f"Shape of train T [BATCH, H]: {t.shape}")
    break
for x,t in test_dataloader:
    print(f"Shape of test X [BATCH, H]: {x.shape}")
    print(f"Shape of test T [BATCH, H]: {t.shape}")
    break
print("--------------end--------------")
print()

class Encoder(nn.Module) :
    def __init__(self, vocab_size, wordvec_size, hidden_size) :
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, wordvec_size)
        self.lstm = nn.LSTM(input_size=wordvec_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, xs) :
        BATCH, SEQ_LEN = xs.shape[0], xs.shape[1]
        emb = self.embedding(xs)
        BATCH, SEQ_LEN, WORDVEC_SIZE = emb.shape[0], emb.shape[1], emb.shape[2]
        y, (hn, cn) = self.lstm(emb)
        LAYER_NUM, BATCH_SIZE, HIDDEN_SIZE = hn.shape[0], hn.shape[1], hn.shape[2]
        return hn

class Decoder(nn.Module) :
    def __init__(self, vocab_size, wordvec_size, hidden_size) :
        super().__init__()
        self.embedding = Embedding(vocab_size, wordvec_size)
        self.lstm = LSTM(wordvec_size, hidden_size, num_layers=1, batch_first=True)
        self.affine = nn.Linear(hidden_size, vocab_size)
    def forward(self, xs, h) :
        BATCH, SEQ_LEN = xs.shape[0], xs.shape[1]
        emb = self.embedding(xs)
        BATCH, SEQ_LEN, WORDVEC_SIZE = emb.shape[0], emb.shape[1], emb.shape[2]
        empty_cn = torch.zeros_like(h)
        y, (hn, cn) = self.lstm(emb, (h, empty_cn))
        y = self.affine(y)
        return y

class Seq2Seq(nn.Module) :
    def __init__(self, vocab_size, wordvec_size, hidden_size) :
        super().__init__()
        self.encoder = Encoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = Decoder(vocab_size, wordvec_size, hidden_size)

    def forward(self, xs, ts) :
        hn = self.encoder(xs)
        y = self.decoder(ts, hn)
        return y

device = 'cuda'
model = Seq2Seq(vocab_size, 128, 256).to(device)
loss_fn = nn.CrossEntropyLoss(reduction='mean')
optimizer = Adam(model.parameters(), lr=0.01)

total_loss = 0.0
total_token = 0
last_loss = 1e5
for epoch in range(max_epoch) :
    for x,t in train_dataloader :
        x,t = x.to(device),t.to(device)
        optimizer.zero_grad()
        y = model(x, t)
        loss = loss_fn(y.reshape(-1, vocab_size), t.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * t.shape[1]
        total_token += t.shape[1]

    avg_loss = total_loss / total_token
    print("epoch:{}, loss:{:.5f}".format(epoch, avg_loss))
    if avg_loss > last_loss or avg_loss < 1e-5 :
        break

with torch.no_grad() :
    test_total_loss = 0.0
    test_total_token = 0
    for x,t in test_dataloader :
        x,t = x.to(device),t.to(device)
        optimizer.zero_grad()
        y = model(x, t)
        loss = loss_fn(y.reshape(-1, vocab_size), t.reshape(-1))
        test_total_loss += loss.detach().item() * t.shape[1]
        test_total_token += t.shape[1]
    test_avg_loss = test_total_loss / test_total_token
    print("test loss:{:.5f}".format(test_avg_loss))