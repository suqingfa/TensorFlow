import tensorflow as tf

(x_train, y_train),(x_test, y_test) =  tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

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

model.compile(optimizer = "adam", 
			  loss = "sparse_categorical_crossentropy", 
			  metrics = ["sparse_categorical_accuracy"])

model.fit(x_train, 
		  y_train, 
		  batch_size = 32, 
		  epochs = 5, 
		  validation_data = (x_test, y_test))

model.summary()