from PIL import Image
import tensorflow as tf
import numpy as np

# 重建神经网络
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

# 加载模型
checkpoint_path = "5_mnist/mnist.ckpt"
model.load_weights(checkpoint_path)

while True:
	# 读入图片
	num = input("input the number of test:")
	img = Image.open("5_mnist/" + num + ".png")
	img = img.resize((28, 28), Image.ANTIALIAS)

	img_arr = np.array(img.convert('L'))
	for i in range(28):
		for j in range(28):
			if img_arr[i][j] < 200:
				img_arr[i][j] = 1
			else:
				img_arr[i][j] = 0

	# 预测
	x_predict = img_arr[tf.newaxis, ...]
	result = model.predict(x_predict)

	# 输出概率
	print(tf.argmax(result, axis = 1))