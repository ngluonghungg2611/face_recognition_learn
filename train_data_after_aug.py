import tensorflow_addons as tfa  
import tensorflow as tf 
from base_network_model import _base_network
from Data_Augumentation import X_au, y_au
model2 = _base_network()

model2.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss = tfa.losses.TripletSemiHardLoss()
)

# Điều chỉnh tăng bacth_size = 64
gen_train2 = tf.data.Dataset.from_tensor_slices((X_au, y_au)).repeat().shuffle(1024).batch(64)
gen_train2
print(len(X_au), len(y_au))
history = model2.fit(
    gen_train2,
    steps_per_epoch = 50,
    epochs=20)
model2.save("model/model_triplot_au.h5")