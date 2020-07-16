from sklearn import datasets
import tensorflow as tf
import numpy as np

# 读入数据
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 乱序
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 数据集分为训练和测试
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 数据类型转换
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 配对并分成batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 超参数
epoch = 500
lr = 0.1

# 参数
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev = 0.1, seed = 1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev = 0.1, seed = 1))

for epoch in range(epoch):

	# 训练
	loss_all = 0
	for step, (x_train, y_train) in enumerate(train_db):
		with tf.GradientTape() as tape:
			y = tf.matmul(x_train, w1) + b1	# Y = X*w1 + b1
			y = tf.math.softmax(y)			# 将Y转换为概率分布
			y_ = tf.one_hot(y_train, 3)		# 将标签转换为独热码
			loss = tf.math.reduce_mean(tf.math.square(y - y_))	# loss_mse
			loss_all += loss.numpy()
		grads = tape.gradient(loss, [w1, b1])# loss对参数求导
		w1.assign_sub(lr * grads[0])		# 反向传播更新参数
		b1.assign_sub(lr * grads[1])
	print(epoch, loss_all / 4)

	# 测试
	total_correct, total_number = 0, 0
	for x_test, y_test in test_db:
		y = tf.matmul(x_test, w1) + b1		# 计算预测Y
		y = tf.math.softmax(y)				# 将Y转换为概率分布
		pred = tf.math.argmax(y, axis = 1)	# 得到概率最大的标签
		pred = tf.cast(pred, y_test.dtype)
		correct = tf.cast(tf.math.equal(pred, y_test), tf.int32)	# 预测值与标准值相同的项，即正确的预测值
		correct = tf.math.reduce_sum(correct)# 正确预测个数
		total_correct += correct.numpy()
		total_number += x_test.shape[0]
	print(total_correct / total_number)		# 预测正确率