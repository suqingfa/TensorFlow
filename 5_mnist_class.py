import tensorflow as tf
import os

(x_train, y_train),(x_test, y_test) =  tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 数据增强
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
generator = tf.keras.preprocessing.image.ImageDataGenerator(
	rotation_range = 45,
	width_shift_range = 0.15,
	height_shift_range = 0.15,
	zoom_range = 0.5
)
generator.fit(x_train)

class MnistModel(tf.keras.Model):
	def __init__(self):
		super(MnistModel, self).__init__()
		self.flatten = tf.keras.layers.Flatten() # 将28*28二维数据拉直为784个数值
		self.d1 = tf.keras.layers.Dense(128, activation = "relu")	# 第一层网络
		self.d2 = tf.keras.layers.Dense(10, activation = "softmax")	# 第二层网络

	def call(self, x):
		y = self.flatten(x)
		y = self.d1(y)
		y = self.d2(y)
		return y

model = MnistModel()

# 读取模型
checkpoint_path = "5_mnist/mnist.ckpt"
if os.path.exists(checkpoint_path + ".index"):
	print("load model")
	model.load_weights(checkpoint_path)

# 保存模型
cp_callback = tf.keras.callbacks.ModelCheckpoint(
	checkpoint_path,
	save_weights_only = True, 
	save_best_only = True
)

model.compile(optimizer = "adam", 
			  loss = "sparse_categorical_crossentropy", 
			  metrics = ["sparse_categorical_accuracy"])

model.fit(generator.flow(x_train, y_train, batch_size = 32), 
		  epochs = 5, 
		  validation_data = (x_test, y_test),
		  callbacks = [cp_callback])

model.summary()

# 显示所有参数
print(model.trainable_variables)