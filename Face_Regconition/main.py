import Pre_train_model
import Cvt_img_blob
from extract_faces import _extract_faces, _extract_bbox, _image_read
import matplotlib.pyplot as plt 
import os
path = "path"
IMAGE_TEST = os.path.join(path, "Dataset/khanh/001.jpg")


image = _image_read(IMAGE_TEST)
bbox = _extract_bbox(image)
face = _extract_faces(image, bbox)
plt.axis("off")
plt.imshow(face)