import pickle
from model_processing import *
def _save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def _load_pickle(file_path):
    with open(file_path, 'wb') as f:
        obj = pickle.load(f)
    return obj
_save_pickle(faces, "./faces.pkl")
_save_pickle(y_labels, "./y_labels.pkl")
_save_pickle(images_file, "./images_file.pkl")