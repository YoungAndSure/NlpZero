#! python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from common.util import CostRecorder, save_model
import time
import torch
import numpy as np
from dataset.ptb import PTBDataset, SequentialBatchSampler
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Embedding
from torch.nn import RNN
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn import init
from dataset.easy_data import HelloDataset
from torch.utils.tensorboard import SummaryWriter

# config:
retrain_and_dump=True
use_ptb=True
seq_len = 50 if use_ptb else 6
batch_size = 8 if use_ptb else 1
max_epoch = 2
file_name = "rnnlm.pth" if use_ptb else "hello.pth"
manual_test_case_size = 10 if use_ptb else 1
write_monitor=False
open_manual_test=False

writer = SummaryWriter(log_dir='rnnlm_monitor') if write_monitor else None

# start
recorder = CostRecorder()

# 读取PTB数据集
train_data = PTBDataset(data_type="train", seq_len=seq_len, cutoff_rate=1) if use_ptb else HelloDataset()
test_data = PTBDataset(data_type="test", seq_len=seq_len, cutoff_rate=1) if use_ptb else HelloDataset()

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

class RnnLm(nn.Module):
    def __init__(self, vocab_size, seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        x_dimention = 128
        hidden_size = 256

        self.embedding = nn.Embedding(vocab_size, x_dimention)

        self.rnn = nn.RNN(input_size=x_dimention, hidden_size=hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:  # 输入到隐藏的权重
                init.xavier_uniform_(param)
            elif 'weight_hh' in name: # 隐藏到隐藏的权重
                init.orthogonal_(param)

        self.affine = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        init.xavier_uniform_(self.affine.weight)

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
    
    # 测试时别忘了reset，不然训练集和测试数据shape不符，隐藏层报错
    def reset_state(self) :
        self.h_last = None

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RnnLm(vocab_size, seq_len).to(device)

# 输入是logits值，目标有两种模式，可以是类别的索引
loss_fn = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

recorder.record("prepare")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof :
#if retrain_and_dump :
    for epoch in range(max_epoch) :
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_token = 0.0
        iter = 0

        model.reset_state() # follow deepseek's advice, really good
        for x,t in train_dataloader :
            x,t = x.to(device), t.to(device)
            y = model.forward(x)
            if (epoch == 0 and iter == 0) :
                print(f"input shape: BATCH, DIMENTION :{x.shape}")
                print(f"output shape: BATCH, SEQ_LEN, DIMENTION :{y.shape}")
                print(f"label shape: BATCH, DIMENTION : {t.shape}")
            # reshape(-1, ?) 表示总数据量不变，根据其他维度推断-1处的位置。
            # 所以这里是把(8, 10, 929589) reshape成了(80, 929589)
            # 这里是把所有batch拍平之后，求的batch_size * seq_len的平均损失
            loss = loss_fn(y.reshape(-1, vocab_size), t.reshape(-1))

            total_loss += loss.detach().item() * x.shape[0] * x.shape[1]
            total_token += x.shape[0] * x.shape[1]

            loss.backward()

            if write_monitor :
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f"{name}_grad", param.grad, epoch)

            optimizer.step()
            optimizer.zero_grad()
            iter += 1
        perplexity = np.exp(total_loss / total_token)
        torch.cuda.synchronize()
        epoch_end = time.perf_counter()
        print("epoch:{}, loss:{:.3f}, perplexity:{:.3f}, cost:{:.3f}ms".format(epoch, total_loss / total_token, perplexity, (epoch_end - epoch_start) * 1000))
    save_model(model, file_name)
prof.export_chrome_trace("rnnlm_profile_maxepoch2.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

recorder.record("train")

if not retrain_and_dump :
    model.load_state_dict(torch.load(file_name))  

model.eval()
with torch.no_grad() :
    total_loss = 0.0
    total_token = 0.0
    model.reset_state()
    for x,t in test_dataloader :
        x,t = x.to(device), t.to(device)
        y = model.forward(x)
        loss = loss_fn(y.reshape(-1, vocab_size), t.reshape(-1))

        total_loss += loss.detach().item() * x.shape[0] * x.shape[1]
        total_token += x.shape[0] * x.shape[1]
    perplexity = np.exp(total_loss / total_token)
    print("test loss:{:.3f}, perplexity:{:.3f}".format(total_loss/total_token, perplexity))

recorder.record("test")
recorder.print_record()

if not open_manual_test :
    exit()
with torch.no_grad() :
    model.reset_state()

    word_to_id, id_to_word = train_data.get_dict()

    inputs = []
    ans = []
    for i in range(manual_test_case_size) :
        ids = train_data.get_random_ids(length=10)
        inputs.append(ids[:-1])
        ans.append(ids)

    x = torch.tensor(np.array(inputs))
    y = model.forward(x.to(device))
    last_word = y[:,-1,:]
    last_word = nn.Softmax(dim=1)(last_word)
    value, idx = last_word.max(dim=1)

    print("\n----------manual test--------")
    for i in range(manual_test_case_size) :
        for word in ans[i] :
            print(id_to_word[word], end=' ')
        print()
        for word in inputs[i] :
            print(id_to_word[word], end=' ')
        print("[{}]".format(id_to_word[idx[i].item()]))
        print()
    print("-----------end------------")