
from torch.utils.data import Dataset
from common.util import *

class HelloW2vDataset(Dataset):
    def __init__(self, window=1):
      text = 'You say goodbye and I say hello'
      corpus, self.word2id, self.id2word = preprocess(text)
      self.corpus = np.array(corpus)
      self.window=window

    def __getitem__(self, index):
      index = index + self.window
      center = self.corpus[index][np.newaxis]
      left_contexts = self.corpus[index-self.window : index]
      right_contexts = self.corpus[index + 1 :index+self.window+1]
      contexts = np.concatenate((left_contexts,right_contexts))
      return contexts, center
    
    def __len__(self):
        return len(self.corpus) - 2 * self.window

    def vocab_size(self) :
       return len(self.corpus)

    def get_dict(self) :
        return self.word2id, self.id2word
    def to_word(self, id) :
      return self.id2word[id]
    def to_words(self, ids) :
      words = []
      for id in ids :
        words.append(self.id2word[id])
      return words
    def to_ids(self, words) :
      ids = []
      for word in words :
        ids.append(self.word2id[word])
      return np.array(ids)

class HelloDataset(Dataset):
    def __init__(self, seq_len=6):
      text = 'You say goodbye and I say hello'
      corpus, self.word2id, self.id2word = preprocess(text)
      self.corpus = np.array(corpus)
      self.seq_len=seq_len

    def __getitem__(self, index):
      start = index * self.seq_len
      end = (index + 1) * self.seq_len
      x = self.corpus[start : end]
      t = self.corpus[start + 1 : end + 1]
      return x, t
    
    def __len__(self):
        return 1

    def vocab_size(self) :
       return len(self.corpus)

    def get_dict(self) :
        return self.word2id, self.id2word

    def get_random_ids(self, length) :
      return self.corpus