#! python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dateutil import parser
from common.util import CostRecorder, save_model
import torch
import numpy as np
from dataset import SequenceDataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import init
from torch.nn import Embedding
from torch.nn import LSTM
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# config:
retrain_and_dump=True
batch_size = 16
max_epoch = 20
file_name = "attention.pth"
manual_test_case_size = 10

# start
recorder = CostRecorder()

train_data = SequenceDataset(data_type="train", data_name="date.txt") 
test_data = SequenceDataset(data_type="test", data_name="date.txt")
vocab_size = train_data.vocab_size()

recorder.record("dataset")

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

recorder.record("dataload")

device = 'cuda'

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

class AttentionEncoder(nn.Module) :
    def __init__(self, vocab_size, wordvec_size, hidden_size) :
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, wordvec_size)
        self.lstm = nn.LSTM(input_size=wordvec_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        init.xavier_uniform_(self.embedding.weight)

    def forward(self, xs) :
        BATCH, SEQ_LEN = xs.shape[0], xs.shape[1]
        emb = self.embedding(xs)
        BATCH, SEQ_LEN, WORDVEC_SIZE = emb.shape[0], emb.shape[1], emb.shape[2]
        y, (hn, cn) = self.lstm(emb)
        LAYER_NUM, BATCH_SIZE, HIDDEN_SIZE = hn.shape[0], hn.shape[1], hn.shape[2]
        return y, hn

class Attention(nn.Module) :
    def __init__(self) :
        super().__init__()

    def forward(self, ys, hs) :
        # 比较蛋疼的就是这个DECODER_SEQ_LEN，干扰了理解
        BATCH, DECODER_SEQ_LEN, HIDDEN_SIZE = ys.shape[0], ys.shape[1], ys.shape[2]
        BATCH, ENCODER_SEQ_LEN, HIDDEN_SIZE = hs.shape[0], hs.shape[1], hs.shape[2]
        hs_for_cal = hs.transpose(1, 2)
        results = torch.matmul(ys, hs_for_cal)
        weights = torch.softmax(results, dim=-1)

        # 看起来注释里的和matmul是等效的
        #hs_unsqueeze = hs.unsqueeze(1).expand(-1, DECODER_SEQ_LEN, -1, -1)
        #weights_broadcast = weights.unsqueeze(3).expand(-1, -1, -1, HIDDEN_SIZE)
        #cal_hs = hs_unsqueeze * weights_broadcast
        #sum_hs = torch.sum(cal_hs, dim=2)
        sum_hs = torch.matmul(weights, hs)

        return sum_hs

class AttentionDecoder(nn.Module) :
    def __init__(self, vocab_size, wordvec_size, hidden_size) :
        super().__init__()
        self.embedding = Embedding(vocab_size, wordvec_size)
        self.lstm = LSTM(wordvec_size, hidden_size, num_layers=1, batch_first=True)
        self.attention = Attention()
        self.affine = nn.Linear(hidden_size, vocab_size)
        init.xavier_uniform_(self.embedding.weight)
        init.xavier_uniform_(self.affine.weight)

    def forward(self, xs, hs, h) :
        BATCH, SEQ_LEN = xs.shape[0], xs.shape[1]
        emb = self.embedding(xs)
        BATCH, SEQ_LEN, WORDVEC_SIZE = emb.shape[0], emb.shape[1], emb.shape[2]

        empty_cn = torch.zeros_like(h)
        ys, (hn, cn) = self.lstm(emb, (h, empty_cn))

        # 编码器的每个时间输出hs,和解码器的每个时间输出ys，计算attention
        y = self.attention(ys, hs)

        y = self.affine(y)
        return y

    def generate(self, xs, hs, h) :
        y = self.forward(xs, hs, h)
        y = torch.argmax(y, dim=2)
        return y

class AttentionSeq2Seq(nn.Module) :
    def __init__(self, vocab_size, wordvec_size, hidden_size) :
        super().__init__()
        self.encoder = AttentionEncoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = AttentionDecoder(vocab_size, wordvec_size, hidden_size)

    def forward(self, xs, ts) :
        ys, hn = self.encoder(xs)
        y = self.decoder(ts, ys, hn)
        return y

    def generate(self, xs, startid, sample_size) :
        BATCH, SEQ_LEN = xs.shape[0], xs.shape[1]
        hs, h = self.encoder(xs)
        ans = torch.zeros(BATCH, 1, dtype=int).to(device)
        ans[:,0] = startid
        while ans.shape[1] < sample_size :
            y = self.decoder.generate(ans, hs, h)
            ans = torch.cat((ans, y[:,-1:]), dim=1)
        return ans

model = AttentionSeq2Seq(vocab_size, 128, 256).to(device)
loss_fn = nn.CrossEntropyLoss(reduction='mean')
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ExponentialLR(optimizer, gamma=0.9)

recorder.record("prepare")

if retrain_and_dump :
    total_loss = 0.0
    total_token = 0
    last_right_count = 0
    right_count_down = 0
    for epoch in range(max_epoch) :
        for x,t in train_dataloader :
            x,t = x.to(device),t.to(device)
            optimizer.zero_grad()
            y = model(torch.flip(x, [1]), t[:,:-1])
            loss = loss_fn(y.reshape(-1, vocab_size), t[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * t.shape[1]
            total_token += t.shape[1]
        scheduler.step()

        avg_loss = total_loss / total_token

        # 测试集计算准确率
        with torch.no_grad() :
            total_count = 0.0
            right_count = 0.0
            startid = 14 # '_'
            for x,t in test_dataloader :
                x,t = x.to(device),t.to(device)
                y = model.generate(torch.flip(x, [1]), startid, 11)
                for i in range(x.shape[0]) :
                    right_ans = train_data.ids_to_string(t[i][1:].detach().to('cpu').numpy())
                    predict_ans = train_data.ids_to_string(y[i][1:].detach().to('cpu').numpy())
                    total_count += 1
                    right_count += 1 if right_ans == predict_ans else 0
            right_rate = right_count / total_count

        print("epoch:{}, loss:{:.5f}, right_count:{}, right_rate:{:.5f}".format(epoch, avg_loss, right_count, right_rate))
        if right_count < last_right_count :
            right_count_down += 1
        if right_count_down > 10 :
            break
        last_right_count = right_count
    save_model(model, file_name)
else :
    model.load_state_dict(torch.load(file_name))

recorder.record("train&test")
recorder.print_record()

with torch.no_grad() :
    char_to_id, id_to_char = train_data.get_vocab()
    startid = char_to_id['_']
    for i in range(manual_test_case_size) :
        question, right_ans = train_data.get_random_case()
        right_ans = train_data.ids_to_string(right_ans)[1:]
        print(train_data.ids_to_string(question).strip(), end='')

        question = torch.tensor(question).to(device)
        predict_ans = model.generate(torch.flip(question.unsqueeze(0), [1]), startid, 11)
        predict_ans = train_data.ids_to_string(predict_ans[0][1:].to('cpu').numpy())
        print("={}".format(predict_ans))
        print("ans:{}  {}".format(right_ans, "x" if right_ans != predict_ans else 'v'))