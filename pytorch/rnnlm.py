#! python3

import torch
from ptb import PTBDataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Embedding
from torch.nn import RNN

seq_len = 10

# 读取PTB数据集
train_data = PTBDataset(data_type = "train", seq_len = seq_len)
test_data = PTBDataset(data_type = "test", seq_len = seq_len)

batch_size = 8

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

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
        x_dimention = 10
        self.embedding = nn.Embedding(vocab_size, x_dimention)
        self.rnn = nn.RNN(input_size=x_dimention, hidden_size=10, num_layers=1, nonlinearity='tanh', batch_first=True)
        #self.affines = [nn.Linear(in_features=x_dimention, out_features=vocab_size) for i in range(seq_len)]
        self.affine = nn.Linear(in_features=seq_len*x_dimention, out_features=seq_len * vocab_size)

    def forward(self, x):
        embs = self.embedding(x)
        BATCH, SEQ_LEN, DIMENTION = embs.shape[0], embs.shape[1], embs.shape[2]
        # 已经设置了batch_first
        # hs是T个RNN的隐藏层输出，h是最后一个rnn的隐藏层输出
        # 为啥输出两个呢？最后一个隐藏层其实包含了前面所有序列的信息，其实是Encode了
        hs,h_last = self.rnn(embs)
        BATCH, SEQ_LEN, DIMENTION = hs.shape[0], hs.shape[1], hs.shape[2]
        ys = self.affine(hs.reshape(BATCH, SEQ_LEN * DIMENTION))
        ys = ys.reshape(BATCH, SEQ_LEN, self.vocab_size)

        return ys
model = RnnLm(len(train_data), seq_len).to(device)

# 输入是logits值，目标有两种模式，可以是类别的索引
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

iter = 0
for x,t in train_dataloader :
    x,t = x.to(device), t.to(device)
    y = model.forward(x)
    if (iter == 0) :
        print(f"input shape: BATCH, DIMENTION :{x.shape}")
        print(f"output shape: BATCH, SEQ_LEN, DIMENTION :{y.shape}")
        print(f"label shape: BATCH, DIMENTION : {t.shape}")
    # reshape(-1, ?) 表示总数据量不变，根据其他维度推断-1处的位置。
    # 所以这里是把(8, 10, 929589) reshape成了(80, 929589)
    loss = loss_fn(y.reshape(-1, len(train_data)), t.reshape(-1))
    loss.backward()
    print("iter:{}, loss:{}".format(iter, loss.data))

    optimizer.step()
    optimizer.zero_grad()
    iter += 1