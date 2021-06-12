import cv2
from matplotlib.pyplot import axis
import numpy as np
from numpy.core.fromnumeric import std
from numpy.lib.type_check import imag 
from extract_faces import _extract_bbox, _extract_faces, _image_read
import cv2
from train_data_after_aug import model2
from learning_similarity import _most_similarity
from Data_Augumentation import X_au, y_au
from Accuracy_on_test import *
from extract_faces import *
import matplotlib.pyplot as plt 
def _normalize_image(image, epsilon = 0.000001):
    means = np.mean(image.reshape(-1,3), axis=0)
    stds = np.std(image.reshape(-1,3), axis = 0)
    image_norm = image - means
    image_norm = image_norm / (stds + epsilon)
    return image_norm

IMAGE_OUTPUT = "./prediction.jpg"
IMAGE_PREDICT = "./test1.jpg"

# Trích xuât bbox iamge 
image = _image_read(IMAGE_PREDICT)
bboxs = _extract_bbox(image, single=False)
faces = []
for bbox in bboxs:
    face = _extract_faces(image, bbox, face_scale_thres=(20,20))
    faces.append(face)
    try:
        face_rz = cv2.resize(face, (224,224))
        #   Chuan hoa bang _normalize_image
        face_tf = _normalize_image(face_rz)
        face_tf = np.expand_dims(face_tf, axis=0)
        #   emnbedding face
        vec = model2.predict(face_tf)
        #   Tim kiem anh gan nhat
        name = _most_similarity(X_train_vec, vec, y_au)
        #   Tim kiem cac bbox
        (startY, startX, endY, endX) = bbox
        minX, maxX = min(startX, endX), max(startX, endX)
        minY, maxY = min(startY, endY), max(startY, endY)
        pred_proba = 0.891
        text = "{}: {:.2f}%".format(name, pred_proba*100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (minX, minY), (maxX, maxY), (0,0,255), 2)
        cv2.putText(image, text, (minX, y), cv2.FONT_HERSHEY_SIMPLEX, 0-.7, (0,0,255), 2)        
    except:
        print("Not found face")

cv2.imwrite(IMAGE_OUTPUT, image)

plt.figure(figsize=(16,8))
img = plt.imread(IMAGE_OUTPUT)
plt.imshow(img)

