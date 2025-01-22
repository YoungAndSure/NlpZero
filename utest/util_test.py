#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
from common.util import *

class UtilTest(unittest.TestCase) :
  def test_preprocess(self) :
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    self.assertEqual(corpus, [0, 1, 2, 3, 4, 1, 5, 6])
    self.assertEqual(word_to_id, {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6})
    self.assertEqual(id_to_word, {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'})

  def test_co_matrix(self) :
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    co_matrix = create_co_matrix(corpus, len(id_to_word), window_size=1)
    self.assertTrue(np.array_equal(co_matrix,[[0,1,0,0,0,0,0],[0,0,1,0,1,1,0],[0,1,0,1,0,0,0],[0,0,1,0,1,0,0],[0,1,0,1,0,0,0],[0,1,0,0,0,0,1],[0,0,0,0,0,1,0]]))
  
  def test_cos_similarity(self) :
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = cos_similarity(x, y)
    self.assertTrue(np.allclose(z, 0.9746318457857057))

  def test_calculate_similary(self) :
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    co_matrix = create_co_matrix(corpus, len(id_to_word), window_size=1)
    similarity = cos_similarity(co_matrix[word_to_id['you']], co_matrix[word_to_id['i']])
    self.assertTrue(np.allclose(similarity, 0.70710677))

  def test_most_similar(self) :
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    co_matrix = create_co_matrix(corpus, len(id_to_word), window_size=1)
    most_similar("You", word_to_id, id_to_word, co_matrix)

unittest.main()