
from torch.utils.data import Dataset
from common.util import *

class HelloW2vDataset(Dataset):
    def __init__(self, window=1, model="cbow"):
      text = 'You say goodbye and I say hello'
      corpus, self.word2id, self.id2word = preprocess(text)
      self.corpus = np.array(corpus)
      self.window=window
      self.model = model

    def __getitem__(self, index):
      index = index + self.window
      left_contexts = self.corpus[index-self.window : index]
      right_contexts = self.corpus[index + 1 :index+self.window+1]
      if self.model == "cbow" :
        contexts = np.concatenate((left_contexts,right_contexts))
        target = self.corpus[index]
      return contexts, target
    
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

class HelloW2vGenerateDataset(Dataset):
    def __init__(self, window=2):
      text = 'You say goodbye and I say hello'
      corpus, self.word2id, self.id2word = preprocess(text)
      self.corpus = np.array(corpus)
      self.window=window
      self.text = self.to_words(self.corpus)

    def __getitem__(self, index):
      index = index + self.window
      contexts = None
      for i in range(self.window) :
        context = self.corpus[index-1-i][np.newaxis]
        contexts = np.concatenate((contexts,context)) if contexts is not None else context
      target = self.corpus[index]
      return contexts, target
    
    def __len__(self):
        return len(self.corpus) - self.window

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



class HelloW2vSkipGramDataset(Dataset):
    def __init__(self, window=1):
      self.text = 'You say goodbye and I say hello'
      corpus, self.word2id, self.id2word = preprocess(self.text)
      self.corpus = np.array(corpus)
      self.window=window

      self.pairs = []
      for idx in range(self.window, len(self.corpus) - self.window):
        center = self.corpus[idx]
        # 左窗口
        for i in range(idx - self.window, idx):
            self.pairs.append((center, self.corpus[i]))
        # 右窗口
        for i in range(idx + 1, idx + self.window + 1):
            self.pairs.append((center, self.corpus[i]))

    def __getitem__(self, index):
      return self.pairs[index]
    
    def __len__(self):
      return len(self.pairs)

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
      if isinstance(words, str) :
        return self.word2id[words]
      ids = []
      for word in words :
        ids.append(self.word2id[word])
      return np.array(ids)
    def get_text(self) :
       return self.text

class HelloW2vWithNegDataset(Dataset):
    def __init__(self, window=1, model="cbow"):
      text = 'You say goodbye and I say hello'
      corpus, self.word2id, self.id2word = preprocess(text)
      self.corpus = np.array(corpus)
      self.window=window
      self.model = model

    def __getitem__(self, index):
      index = index + self.window
      left_contexts = self.corpus[index-self.window : index]
      right_contexts = self.corpus[index + 1 :index+self.window+1]
      if self.model == "cbow" :
        contexts = np.concatenate((left_contexts,right_contexts))
        target = self.corpus[index]
      return contexts, target
    
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