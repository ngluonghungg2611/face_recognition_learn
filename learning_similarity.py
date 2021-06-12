from numpy.core.fromnumeric import argmax
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from share_data80_20 import X_test, X_train, y_train, y_test

def _most_similarity(embed_vecs, vec, labels):
  sim = cosine_similarity(embed_vecs, vec)
  sim = np.squeeze(sim, axis = 1)
  argmax = np.argsort(sim)[::-1][:1]
  label = [labels[idx] for idx in argmax][0]
  return label

#Lay ngau nhien buc anh trong dataset
vec = X_test[1].reshape(1, -1)