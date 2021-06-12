from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Preprocessing_data import X_train, X_test, y_train, y_test
import numpy as np 


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

#   Với mỗi bức ảnh trên tập train sẽ lấy ra 5 ảnh biến thể --> Như vậy 
# ta có gần 550 ảnh 
no_batch = 0
X_au = []
y_au = []
for i in np.arange(len(X_train)):
    no_img = 0
    for x in datagen.flow(np.expand_dims(X_train[i], axis=0), batch_size=1):
        X_au.append(x[0])
        y_au.append(y_train[i])
        no_img += 1
        if no_img == 5:
            break