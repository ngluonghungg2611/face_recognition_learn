from os import name
from posixpath import split
from imutils import paths  
DATASET_PATH = "./path/Dataset"
from extract_faces import _extract_bbox, _extract_faces, _image_read

def _model_processing(face_scale_thres = (20,20)):
    """
        face_scale_thres: Ngưỡng (w,h) để chấp nhận 1 khuôn mặt 
    """
    
    image_links = list(paths.list_images(DATASET_PATH))
    image_file = []
    y_labels = []
    faces = []
    total = 0
    for image_link in image_links:
        split_image_links = image_link.split("/")
        
        # put label
        name = split_image_links[-2]
        
        #   read image
        image = _image_read(image_link)
        (h, w) = image.shape[:2]
        
        #Dectect các vị trí khuôn mặt. Giả định rằng mỗi bức ảnh chỉ có 1 khôn mặt 
        bbox = _extract_bbox(image, single=True)
        if bbox is not None:
            # Lay ra face
            face = _extract_faces(image, bbox, face_scale_thres=(20,20))
            if face is not None:
                faces.append(face)
                y_labels.append(name)
                image_file.append(image_links)
                total += 1
            else:
                next
    print("Total bbox face extracted: {}".format(total))
    return faces, y_labels, image_file

faces, y_labels, images_file = _model_processing()