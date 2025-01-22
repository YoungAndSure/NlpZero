#! python3

def preprocess(text) :
  text = text.lower()
  text = text.replace('.', ' .')
  words = text.split(' ')

  word2id={}
  id2word={}
  for word in words :
    if word not in word2id :
      new_id = len(word2id)
      word2id[word] = new_id
      id2word[new_id] = word

  corpus = [word2id[w] for w in words]

  return corpus, word2id, id2word
