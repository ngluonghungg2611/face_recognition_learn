import tensorflow_addons as tfa
import tensorflow as tf 
from base_network_model import model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tfa.losses.TripletSemiHardLoss())