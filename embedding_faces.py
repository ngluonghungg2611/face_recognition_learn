from Pre_train_model import encoder
from Cvt_img_blob import _blobimage
from extract_faces import _extract_bbox, _extract_faces, _image_read
from model_processing import faces, y_labels, images_file
from save_data import _load_pickle, _save_pickle
def _embedding_faces(encoder, faces):
    emb_vecs = []
    for face in faces:
        faceBlob = _blobimage(face, out_size=(96, 96), scalefactor=1/255.0, mean=(0, 0, 0) )
        #Embedding face
        encoder.setInput(faceBlob)
        vec = encoder.forward()
        emb_vecs.append(vec)
    return emb_vecs
embed_faces = _embedding_faces(encoder, faces)
# Nhớ save embed_faces vào Dataset.
_save_pickle(embed_faces, "./embed_blob_faces.pkl")
        