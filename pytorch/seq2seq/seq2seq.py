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

# config:
retrain_and_dump=False
batch_size = 8
max_epoch = 100
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
    break
for x,t in test_dataloader:
    print(f"Shape of test X [BATCH, H]: {x.shape}")
    break
print("--------------end--------------")
print()