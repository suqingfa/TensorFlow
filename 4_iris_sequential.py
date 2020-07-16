from sklearn import datasets
import tensorflow as tf

# 训练数据
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

x_data = tf.random.shuffle(x_data, seed = 116)
y_data = tf.random.shuffle(y_data, seed = 116)

# 定义网络结构
model = tf.keras.Sequential([
	tf.keras.layers.Dense(3, activation = "softmax", kernel_regularizer = tf.keras.regularizers.l2())
])

# 训练方法
model.compile(optimizer = "sgd", loss = "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])

# 执行训练过程
model.fit(x_data, y_data, batch_size = 32, epochs = 500, validation_split = 0.2, validation_freq = 20)

# 打印网络结构
model.summary()