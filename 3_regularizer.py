import tensorflow as tf
from sklearn import datasets

# 读入数据集
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

x_data = tf.random.shuffle(x_data, seed = 116)
y_data = tf.random.shuffle(y_data, seed = 116)

x_data = tf.cast(x_data, tf.float32)

# 将数据集分为训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 参数
lr = 0.1
epoch = 500
REGULARIZER = 0.03

w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev = 0.1, seed = 1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev = 0.1, seed = 1))

for epoch in range(epoch):

	# 训练
	loss_all = 0
	for x_train, y_train in train_db:
		with tf.GradientTape() as tape:
			y = tf.matmul(x_train, w1) + b1
			y = tf.math.softmax(y)
			y_ = tf.one_hot(y_train, 3)
			# 正则化
			loss = tf.math.reduce_mean(tf.math.square(y - y_)) + REGULARIZER * tf.nn.l2_loss(w1)
			loss_all += loss.numpy()
		grads = tape.gradient(loss, [w1, b1])
		w1.assign_sub(lr * grads[0])
		b1.assign_sub(lr * grads[1])
	print(epoch, loss_all / 4)

	# 测试
	total_correct, total_number = 0, 0
	for x_test, y_test in test_db:
		y = tf.matmul(x_test, w1) + b1
		pred = tf.math.argmax(y, axis = 1)
		pred = tf.cast(pred, y_test.dtype)
		correct = tf.cast(tf.equal(y_test, pred), tf.float32)
		total_correct += tf.math.reduce_sum(correct)
		total_number += x_test.shape[0]
	print(total_correct.numpy() / total_number)