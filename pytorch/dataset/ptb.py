# coding: utf-8
import sys
import os
sys.path.append('..')
try:
    import urllib.request
except ImportError:
    raise ImportError('Use Python3!')
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler

url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
key_file = {
    'train':'ptb.train.txt',
    'test':'ptb.test.txt',
    'valid':'ptb.valid.txt'
}
save_file = {
    'train':'ptb.train.npy',
    'test':'ptb.test.npy',
    'valid':'ptb.valid.npy'
}
vocab_file = 'ptb.vocab.pkl'

dataset_dir = os.path.dirname(os.path.abspath(__file__)) + '/../data'


def _download(file_name):
    file_path = dataset_dir + '/' + file_name
    if os.path.exists(file_path):
        return

    print('Downloading ' + file_name + ' ... ')

    try:
        urllib.request.urlretrieve(url_base + file_name, file_path)
    except urllib.error.URLError:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url_base + file_name, file_path)

    print('Done')


def load_vocab():
    vocab_path = dataset_dir + '/' + vocab_file

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}
    data_type = 'train'
    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name

    _download(file_name)

    words = open(file_path).read().replace('\n', '<eos>').strip().split()

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word


def load_data(data_type='train', cutoff_rate=1.0):
    '''
        :param data_type: 数据的种类：'train' or 'test' or 'valid (val)'
        :return:
    '''
    if data_type == 'val': data_type = 'valid'
    save_path = dataset_dir + '/' + save_file[data_type]

    word_to_id, id_to_word = load_vocab()

    if os.path.exists(save_path):
        corpus = np.load(save_path)
        corpus = corpus[:int(corpus.shape[0] * cutoff_rate)]
        return corpus, word_to_id, id_to_word

    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name
    _download(file_name)

    words = open(file_path).read().replace('\n', '<eos>').strip().split()
    corpus = np.array([word_to_id[w] for w in words])

    np.save(save_path, corpus)
    corpus = corpus[:int(corpus.shape[0] * cutoff_rate)]
    return corpus, word_to_id, id_to_word

class PTBDataset(Dataset):
    def __init__(self, data_type, seq_len=1, cutoff_rate=1.0):
        self.seq_len = seq_len
        self.corpus, self.word_to_id, self.id_to_word = self._process_file(data_type, cutoff_rate)
    
    def _process_file(self, data_type, cutoff_rate):
        return load_data(data_type, cutoff_rate)
    
    def __getitem__(self, index):
        x_start = index * self.seq_len
        x_end = x_start + self.seq_len

        t_start = x_start + 1
        t_end = t_start + self.seq_len

        return self.corpus[x_start : x_end], self.corpus[t_start : t_end]
    
    def __len__(self):
        # 不然到最后会因为x和t数量不一致而挂掉
        return len(self.corpus) // self.seq_len
    
    def vocab_size(self) :
        return len(self.word_to_id)

    def getword(self, id) :
        return self.id_to_word[id]

    def get_dict(self) :
        return self.word_to_id, self.id_to_word
    
    def get_random_ids(self, length=20) :
        start = np.random.randint(0, len(self.corpus) - 20)
        # 24 is <eof>
        while self.corpus[start] != 24 :
            start += 1
        start += 1
        # return answer, so +1
        ids = self.corpus[start : start + length + 1]
        return ids

class SequentialBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        num_batches = len(self.dataset) // self.batch_size
        for i in range(num_batches):
            batch_index = []
            for j in range(self.batch_size) :
                batch_index.append(i + j * num_batches)
            yield batch_index

    def __len__(self):
        return len(self.dataset) // self.batch_size