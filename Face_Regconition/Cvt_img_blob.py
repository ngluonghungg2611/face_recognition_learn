import cv2

def _blobimage(image, out_size = (300,300), scaleFactor = 1.0,
               mean = (104.0, 177.0, 123.0)):
    """
    input: 
        image: ma tran RGB cua input image
        out_size: Kich thuoc cua anh blob
    output:
        imageBlob: anh blob
    """
    # chuyển sang blob image để tránh bị nhiễu sáng 
    imageBlob = cv2.dnn.blobFromImage(image,
                                      scaleFactor = scaleFactor, #scale image
                                      size = out_size, #Output shape
                                      mean = mean, # Trung binh theo kenh RGB
                                      swapRB = False, # Truong hopanh sang la BGR thi xet bang True de chuyen sang RGB
                                      crop = False)
    return imageBlob
