#! python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from basic_dataset.sequence import *
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, data_type='train', data_name='addition.txt', add_eos=False):
      (x_train, t_train), (x_test, t_test) = load_data(data_name, add_eos=add_eos)
      if data_type == 'train' :
         self.xs = x_train
         self.ts = t_train
      else :
         self.xs = x_test
         self.ts = t_test
      
      self.char_to_id, self.id_to_char = get_vocab()

    def __getitem__(self, index):
      return self.xs[index], self.ts[index]
    
    def __len__(self):
        return len(self.xs)

    def vocab_size(self) :
       return len(self.char_to_id)

    def get_vocab(self) :
        return self.char_to_id, self.id_to_char

    def get_random_case(self) :
      index = np.random.randint(0, len(self.xs))
      return self.xs[index], self.ts[index]

    def c2i(self, id) :
      return self.char_to_id[id]

    def ids_to_string(self, ids, d='') :
      string = []
      for i in range(ids.shape[0]) :
         string.append(self.id_to_char[ids[i]])
      return d.join(string)