#! python3

import torch
from ptb import PTBDataset
from torch.utils.data import DataLoader

# 读取PTB数据集
train_data = PTBDataset("train")
test_data = PTBDataset("test")

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {x.shape}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

