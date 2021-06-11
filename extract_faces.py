from face_recognition import face_locations
import matplotlib.pyplot as plt 
import cv2
import os
path = "path"
IMAGE_TEST = os.path.join(path, "Dataset/khanh/001.jpg")

def _image_read(image_path):
    """
    input: 
        image_path: link file image
    return:
        image: np.arrayof image
    """
    image= cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

image = _image_read(IMAGE_TEST)
# cv2.imshow("test", image)
# cv2.waitKey(0)

def _extract_bbox(image, single = True):
    """
    Trích xuất ra tọa độ các face từ ảnh input
    input:
        image: anh input theo RGB
        single: Lấy ra 1 face trên 1 bức ảnh nếu True hoặc nhiều False nếu False. Mặc định là true
    return:
        bbox: Toa do cua  bbox: <startY>, <startX>, <endY>, <endX>
    """
    bboxs = face_locations(image)
    if len(bboxs) == 0:
        return None
    if single:
        bbox = bboxs[0]
        return bbox
    else: 
        return bboxs
    
def _extract_faces(image, bbox, face_scale_thres = (20,20)):
    """
    input:
        image: ma trận RGB ảnh đầu vào
        bbox: tọa độ của ảnh input
        face_scale_thres: ngưỡng kích thước (h, w) của face. Nếu nhỏ hơn ngưỡng này thì loại bỏ face
    return:
        face: ma trận RGB ảnh khuôn mặt được trích xuất từ image input.
  """
    h,w = image.shape[:2]
    try:
        (startY, startX, endY, endX) = bbox
    except:
        return None

    minX, maxX = min(startX, endX), max(startX, endX)
    minY, maxY = min(startY, endY), max(startY, endY)
    face = image[minY:maxY, minX:maxX].copy()
    
    # extract the face ROI and grab the ROI dimensions
    (fH, fW) = face.shape[:2]
        
    # ensure the face width and height are sufficiently large
    if fW < face_scale_thres[0] or fH < face_scale_thres[1]:
        return None
    else:
        return face
        
