#! python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
from torch import nn
import numpy as np
import math
from dataset import RedgptDataset
from torch.utils.data import DataLoader

batch_size = 32

data_path = "../../../RedGPT/RedGPT-Dataset-V1-CN.json"
train_data = RedgptDataset(data_path, data_type="train", add_eos=True) 
test_data = RedgptDataset(data_path, data_type="test", add_eos=True)
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