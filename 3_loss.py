import tensorflow as tf
import numpy as np

# 训练数据
rdm = np.random.RandomState(23455)
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + rdm.rand() / 10 - 0.05] for (x1, x2) in x]
x = tf.cast(x, tf.float32)

# 参数
lr = 0.002
epoch = 15000
w1 = tf.Variable(tf.random.normal([2, 1], stddev = 1, seed = 1))

def loss_fn(y, y_):
	# return tf.math.reduce_mean(tf.math.square(y - y_)) # loss_mse
	# return tf.math.reduce_mean(tf.where(tf.greater(y, y_), 1 * (y - y_), 99 * (y_ - y))) # 自定义损失函数
	return tf.losses.categorical_crossentropy(y_, y)	# 交叉熵损失函数

for epoch in range(epoch):
	with tf.GradientTape() as tape:
		y = tf.matmul(x, w1)
		loss = loss_fn(y, y_)
	grads = tape.gradient(loss, w1)
	w1.assign_sub(lr * grads)

	if epoch % 500 == 0:
		print(epoch, w1.numpy())

print(w1.numpy())