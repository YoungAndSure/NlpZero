#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.util import *
import matplotlib.pyplot as plt
import dataset.ptb as ptb
from sklearn.utils.extmath import randomized_svd

window_size=2
wordvec_size=100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
co_matrix = create_co_matrix(corpus, vocab_size, window_size=window_size)
C = ppmi(co_matrix)

U,S,V = randomized_svd(C, n_components=wordvec_size, n_iter=5, random_state=True)
print(U.shape)