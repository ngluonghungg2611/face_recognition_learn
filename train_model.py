import tensorflow as tf 
from share_data80_20 import y_train, y_test
from Preprocessing_data import X_train
from base_network_model import model
print(X_train.shape, len(y_train))

gen_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(1024).batch(32)
gen_train
history = model.fit(
    gen_train,
    steps_per_epoch = 50,
    epochs=10)