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

unittest.main()