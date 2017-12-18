import numpy as np
from numpy import linalg as LA
import pickle
from collections import Counter
import csv

class Vocabulary(object):
  def __init__(self, vocab_file, emb_file='', dim_emb=0):
    with open(vocab_file, 'rb') as f:
        self.size, self.word2id, self.id2word = pickle.load(f)
    self.dim_emb = dim_emb
    self.embedding = np.random.random_sample(
        (self.size, self.dim_emb)) - 0.5

    if emb_file:
        with open(emb_file) as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]])
                if word in self.word2id:
                    self.embedding[self.word2id[word]] = vec

    for i in range(self.size):
        self.embedding[i] /= LA.norm(self.embedding[i])

def build_vocab(data, vocab_path, vocab_metadata_path, min_occur=5):

    word2id = {'<pad>':0, '<go>':1, '<eos>':2, '<unk>':3}
    id2word = ['<pad>', '<go>', '<eos>', '<unk>']

    words = [word for sent in data for word in sent]
    cnt = Counter(words)
    for word in cnt:
        if cnt[word] >= min_occur:
            word2id[word] = len(word2id)
            id2word.append(word)
    vocab_size = len(word2id)
    with open(vocab_path, 'wb') as f:
        pickle.dump((vocab_size, word2id, id2word), f, pickle.HIGHEST_PROTOCOL)

    """Writes metadata file for Tensorboard word embedding visualizer as described here:
      https://www.tensorflow.org/get_started/embedding_viz
    """
    print("Writing word embedding metadata file to %s" % (vocab_metadata_path))
    with open(vocab_metadata_path, "w") as f:
        fieldnames = ['word']
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        for w in id2word:
            writer.writerow({"word": w})