import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class CifarModel(Model):
	def __init__(self):
		super(CifarModel, self).__init__()

		# 卷积 CBAPD
		self.c1 = Conv2D(filters = 6, kernel_size = (5, 5), padding = "same")
		self.b1 = BatchNormalization()
		self.a1 = Activation("relu")
		self.p1 = MaxPool2D(strides = 2, padding = "same")
		self.d1 = Dropout(0.2)

		self.flatten = Flatten()
		self.f1 = Dense(128, activation = "relu")
		self.d2 = Dropout(0.2)
		self.f2 = Dense(10, activation = "softmax")

	def call(self, x):
		x = self.c1(x)
		x = self.b1(x)
		x = self.a1(x)
		x = self.p1(x)
		x = self.d1(x)

		x = self.flatten(x)
		x = self.f1(x)
		x = self.d2(x)
		x = self.f2(x)

		return x

model = CifarModel()

model.compile(
	optimizer = "adam",
	loss = "sparse_categorical_crossentropy",
	metrics = ["sparse_categorical_accuracy"]
)

model.fit(
	x_train,
	y_train,
	batch_size = 32,
	epochs = 5,
	validation_data = (x_test, y_test)
)

model.summary()