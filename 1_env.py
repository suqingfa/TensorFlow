import tensorflow as tf

version = tf.__version__
devices = tf.config.list_physical_devices()

print(version, devices)