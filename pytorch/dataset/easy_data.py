
from torch.utils.data import Dataset
from common.util import *

class EasyDataset(Dataset):
    def __init__(self, window_size=1, enable_unigram_sampler=True):
      text = 'You say goodbye and I say hello.'
      self.corpus, word2id, id2word = preprocess(text)
      contexts, targets = create_contexts_target(corpus=self.corpus, window_size=window_size)
      self.contexts, self.targets = np.array(contexts), np.array(targets)
      self.enable_unigram_sampler = enable_unigram_sampler

      # 初始化负样本采样
      if self.enable_unigram_sampler :
        self.unigram_sampler = UnigramSampler(self.corpus, 0.75, 5)
    
    def __getitem__(self, index):
      input = self.contexts[index, :]
      positive_targets = self.targets[index, :]
      positive_labels = np.ones_like(positive_targets, dtype=np.float32)

      if self.enable_unigram_sampler :
        negative_targets = self.unigram_sampler.get_negative_sample(positive_targets)
        negative_labels = np.zeros_like(negative_targets, dtype=np.float32)
        targets = np.concatenate((positive_targets[np.newaxis, :], negative_targets), axis=1)
        labels = np.concatenate((positive_labels[np.newaxis, :], negative_labels), axis=1)
      else :
        targets = positive_targets
        labels = positive_labels

      return input, targets.squeeze(), labels
    
    def __len__(self):
        return self.targets.shape[0]

    def vocab_size(self) :
       return len(self.corpus)

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