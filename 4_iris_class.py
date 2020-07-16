import tensorflow as tf
from sklearn import datasets

# 训练数据
x_data = tf.random.shuffle(datasets.load_iris().data, seed = 116)
y_data = tf.random.shuffle(datasets.load_iris().target, seed = 116)

# 定义网络结构
class IrisModel(tf.keras.Model):
	def __init__(self):
		super(IrisModel, self).__init__()
		self.d1 = tf.keras.layers.Dense(3, activation = "softmax", kernel_regularizer = tf.keras.regularizers.l2())

	def call(self, x):
		return self.d1(x)

model = IrisModel()

# 训练方法
model.compile(optimizer = "sgd", loss = "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])

# 执行训练过程
model.fit(x_data, y_data, batch_size = 32, epochs = 500, validation_split = 0.2, validation_freq = 20)

# 打印网络结构
model.summary()