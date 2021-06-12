from save_data import _load_pickle, _save_pickle
import cv2
import numpy as np 
from share_data80_20 import X_test, X_train, y_train, y_test
faces = _load_pickle("faces.pkl")

faceResize = []
for face in faces:
    face_rz = cv2.resize(face, (224,224))
    faceResize.append(face_rz)
    
X = np.stack(faceResize)
X.shape
#   Phan chia tap train/test
id_train = _load_pickle("./id_train.pkl")
id_test = _load_pickle("./id_test.pkl")

X_train, X_test = X[id_train], X[id_test]

print(X_train.shape)
print(X_test.shape)
