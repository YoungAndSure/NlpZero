#! python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
from common.util import CostRecorder
import torch
from ptb import PTBDataset, SequentialBatchSampler
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Embedding
from torch.nn import RNN
from torch.profiler import profile, record_function, ProfilerActivity

seq_len = 50
batch_size = 8
max_epoch = 10

recorder = CostRecorder()

# 读取PTB数据集
train_data = PTBDataset(data_type="train", seq_len=seq_len, cutoff_rate=1)
test_data = PTBDataset(data_type="test", seq_len=seq_len, cutoff_rate=1)
# 测试集的词汇表是训练集的子集
vocab_size = train_data.vocab_size()

recorder.record("dataset")

# Create data loaders.
train_batch_sampler = SequentialBatchSampler(train_data, batch_size)
train_dataloader = DataLoader(train_data,
                              batch_sampler=train_batch_sampler)
test_batch_sampler = SequentialBatchSampler(test_data, batch_size)
test_dataloader = DataLoader(test_data,
                             batch_sampler=test_batch_sampler)

recorder.record("dataload")

print("vocab_size:", vocab_size)
print("train data size:", len(train_dataloader) * batch_size)
print("test data size", len(test_dataloader) * batch_size)
for x,t in train_dataloader:
    print(f"Shape of train X [BATCH, H]: {x.shape}")
    break
for x,t in test_dataloader:
    print(f"Shape of test X [BATCH, H]: {x.shape}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class RnnLm(nn.Module):
    def __init__(self, vocab_size, seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        x_dimention = 128
        hidden_size = 256
        self.embedding = nn.Embedding(vocab_size, x_dimention)
        self.rnn = nn.RNN(input_size=x_dimention, hidden_size=hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True)
        self.affine = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.h_last = None

    def forward(self, x):
        embs = self.embedding(x)
        BATCH, SEQ_LEN, DIMENTION = embs.shape[0], embs.shape[1], embs.shape[2]
        # 已经设置了batch_first
        # hs是T个RNN的隐藏层输出，h是最后一个rnn的隐藏层输出
        # 为啥输出两个呢？最后一个隐藏层其实包含了前面所有序列的信息，其实是Encode了
        if self.h_last is not None:
            self.h_last = self.h_last.detach()
        hs,self.h_last = self.rnn(embs) if self.h_last is None else self.rnn(embs, self.h_last)
        BATCH, SEQ_LEN, DIMENTION = hs.shape[0], hs.shape[1], hs.shape[2]
        ys = self.affine(hs)
        return ys

model = RnnLm(vocab_size, seq_len).to(device)

# 输入是logits值，目标有两种模式，可以是类别的索引
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

recorder.record("prepare")

#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof :
if True :
    for epoch in range(max_epoch) :
        total_loss = 0.0
        iter = 0
        for x,t in train_dataloader :
            x,t = x.to(device), t.to(device)
            y = model.forward(x)
            if (epoch == 0 and iter == 0) :
                print(f"input shape: BATCH, DIMENTION :{x.shape}")
                print(f"output shape: BATCH, SEQ_LEN, DIMENTION :{y.shape}")
                print(f"label shape: BATCH, DIMENTION : {t.shape}")
            # reshape(-1, ?) 表示总数据量不变，根据其他维度推断-1处的位置。
            # 所以这里是把(8, 10, 929589) reshape成了(80, 929589)
            loss = loss_fn(y.reshape(-1, vocab_size), t.reshape(-1))
            total_loss += loss.data
            if (iter % 500 == 0) :
                print("epoch:{}, iter:{}, loss:{}".format(epoch, iter, total_loss / (iter + 1)))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            iter += 1
        print("epoch:{}, loss:{}".format(epoch, total_loss / iter))
#prof.export_chrome_trace("rnnlm_profile.json")
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

recorder.record("train")

with torch.no_grad() :
    total_loss = 0.0
    for x,t in test_dataloader :
        x,t = x.to(device), t.to(device)
        y = model.forward(x)
        loss = loss_fn(y.reshape(-1, vocab_size), t.reshape(-1))
        total_loss += loss.data
    avg_loss = total_loss / len(test_dataloader)
    print("test avg_loss:{}".format(avg_loss))

recorder.record("test")
recorder.print_record()