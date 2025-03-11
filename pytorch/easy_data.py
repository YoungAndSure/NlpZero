
from torch.utils.data import Dataset
from common.util import *

class EasyDataset(Dataset):
    def __init__(self, window_size=1):
      text = 'You say goodbye and I say hello.'
      self.corpus, word2id, id2word = preprocess(text)
      self.contexts, self.targets = create_contexts_target(corpus=self.corpus, window_size=window_size)
      print(self.contexts.shape, self.targets.shape)

      # 初始化负样本采样
      self.unigram_sampler = UnigramSampler(self.corpus, 0.75, 5)
    
    def __getitem__(self, index):
      input = self.contexts[index, :]
      positive_targets = self.targets[index, :]
      negative_targets = self.unigram_sampler.get_negative_sample(positive_targets)
      targets = np.concatenate((positive_targets[np.newaxis, :], negative_targets), axis=1)

      positive_labels = np.ones_like(positive_targets, dtype=np.float32)
      negative_labels = np.zeros_like(negative_targets, dtype=np.float32)
      labels = np.concatenate((positive_labels[np.newaxis, :], negative_labels), axis=1)
      return input, targets.squeeze(), labels
    
    def __len__(self):
        return self.targets.shape[0]

    def vocab_size(self) :
       return len(self.corpus)