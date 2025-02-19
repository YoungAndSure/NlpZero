#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.util import *
import matplotlib.pyplot as plt

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
co_matrix = create_co_matrix(corpus, len(id_to_word), window_size=1)
M = ppmi(co_matrix)
U,S,V = np.linalg.svd(M)

for word, id in word_to_id.items() :
  plt.annotate(word, (U[id][0], U[id][1]))
plt.scatter(U[:][0], U[:][1])