from save_data import _load_pickle, _save_pickle
import embedding_faces
import model_processing
import numpy as np 
from sklearn.model_selection import train_test_split
embed_faces = _load_pickle("./embed_blob_faces.pkl")
y_labels = _load_pickle("./y_labels.pkl")

ids = np.arange(len(y_labels))

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(np.stack(embed_faces), y_labels, ids, test_size = 0.2, stratify = y_labels)
X_train = np.squeeze(X_train, axis = 1)
X_test = np.squeeze(X_test, axis = 1)
print(X_train.shape, X_test.shape)
print(len(y_train), len(y_test))       

_save_pickle(id_train, "./id_train.pkl")
_save_pickle(id_test, "./id_test.pkl")