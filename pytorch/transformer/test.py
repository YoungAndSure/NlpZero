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
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from transformer import Transformer

# config:
retrain_and_dump=True
batch_size = 1
max_epoch = 1
file_name = "transformer.pth"
manual_test_case_size = 10

# start
recorder = CostRecorder()

train_data = SequenceDataset(data_type="train", data_name="date.txt", add_eos=True) 
test_data = SequenceDataset(data_type="test", data_name="date.txt", add_eos=True)
vocab_size = train_data.vocab_size()

startid = train_data.c2i('_')
endid = train_data.c2i('.')

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

seq_len = 3
d_model = 64
dim_feedforward = d_model * 4
nhead = 8
batch_size = 2
encoder_layer = 6
decoder_layer = 6
model = Transformer(vocab_size, d_model, nhead, dim_feedforward, encoder_layer, decoder_layer).to(device)
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
            # september 27, 1994           _1994-09-27.
            # x = "september 27, 1994           "
            # t = "_1994-09-27"
            y = model(x, t[:, :-1])
            # label:1994-09-27.
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
            for x,t in test_dataloader :
                x,t = x.to(device),t.to(device)
                y = model.generate(x, startid, 12, endid)
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
        predict_ans = model.generate(question.unsqueeze(0), startid, 12, endid)
        predict_ans = train_data.ids_to_string(predict_ans[0][1:].to('cpu').numpy())
        print("={}".format(predict_ans))
        print("ans:{}  {}".format(right_ans, "x" if right_ans != predict_ans else 'v'))