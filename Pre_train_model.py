#Ham load model
## Ham load model tu caffe

import cv2
import os
import numpy as np  
path = "./path"
EMBEDDING_FL = os.path.join(path, "nn4.small2.v1.t7") 
DATASET_PATH = os.path.join(path, "Dataset")

def _load_model(model_path_fl):
    """
        model_path_fl: link file chua weight cua model
    """
    model = cv2.dnn.readNetFromTorch(model_path_fl)
    return model

encoder = _load_model(EMBEDDING_FL)
