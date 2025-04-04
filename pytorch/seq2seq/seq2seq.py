#! python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

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
max_epoch = 200
file_name = "seq2seq.pth"
manual_test_case_size = 10

# start
recorder = CostRecorder()

train_data = SequenceDataset(data_type="train") 
test_data = SequenceDataset(data_type="test")
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

class Encoder(nn.Module) :
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
        return hn

class Decoder(nn.Module) :
    def __init__(self, vocab_size, wordvec_size, hidden_size) :
        super().__init__()
        self.embedding = Embedding(vocab_size, wordvec_size)
        self.lstm = LSTM(wordvec_size + hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.affine = nn.Linear(hidden_size + hidden_size, vocab_size)
        init.xavier_uniform_(self.embedding.weight)
        init.xavier_uniform_(self.affine.weight)

    def forward(self, xs, h) :
        BATCH, SEQ_LEN = xs.shape[0], xs.shape[1]
        emb = self.embedding(xs)
        BATCH, SEQ_LEN, WORDVEC_SIZE = emb.shape[0], emb.shape[1], emb.shape[2]

        h_for_cat = h.transpose(1,0).expand(-1,emb.shape[1],-1)
        emb_cat_h = torch.concat((emb, h_for_cat), dim=2)
        empty_cn = torch.zeros_like(h)
        y, (hn, cn) = self.lstm(emb_cat_h, (h, empty_cn))

        y_cat_h = torch.concat((y, h_for_cat), dim=2)
        y = self.affine(y_cat_h)
        return y

    def generate(self, xs, h) :
        y = self.forward(xs, h)
        y = torch.argmax(y, dim=2)
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

    def generate(self, xs, startid, sample_size) :
        BATCH, SEQ_LEN = xs.shape[0], xs.shape[1]
        hn = self.encoder(xs)
        ans = torch.zeros(BATCH, 1, dtype=int).to(device)
        ans[:,0] = startid
        while ans.shape[1] < sample_size :
            y = self.decoder.generate(ans, hn)
            ans = torch.cat((ans, y[:,-1:]), dim=1)
        return ans

model = Seq2Seq(vocab_size, 128, 256).to(device)
loss_fn = nn.CrossEntropyLoss(reduction='mean')
optimizer = Adam(model.parameters(), lr=0.01)
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
            startid = 6 # '_'
            for x,t in test_dataloader :
                x,t = x.to(device),t.to(device)
                y = model.generate(torch.flip(x, [1]), startid, 5)
                for i in range(x.shape[0]) :
                    right_ans = eval(train_data.ids_to_string(x[i].detach().to('cpu').numpy()))
                    predict_ans = int(train_data.ids_to_string(y[i][1:].detach().to('cpu').numpy()))
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

recorder.record("train&text")
recorder.print_record()

with torch.no_grad() :
    char_to_id, id_to_char = train_data.get_vocab()
    startid = char_to_id['_']
    for i in range(manual_test_case_size) :
        question, t = train_data.get_random_case()
        right_ans = eval(train_data.ids_to_string(question))
        print(train_data.ids_to_string(question).strip(), end='')

        question = torch.tensor(question).to(device)
        predict_ans = model.generate(torch.flip(question.unsqueeze(0), [1]), startid, 5)
        predict_ans = train_data.ids_to_string(predict_ans[0][1:].to('cpu').numpy())
        print("={}".format(predict_ans))
        print("ans:{}  {}".format(right_ans, "x" if right_ans != int(predict_ans) else 'v'))