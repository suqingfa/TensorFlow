import tensorflow as tf

w = tf.Variable(tf.constant(-5.0))
lr = 0.2
epoch = 40

for epoch in range(epoch):
	with tf.GradientTape() as tape:	# 梯度计算过程
		loss = tf.square(w+1)		# loss = (w+1)^2
	grads = tape.gradient(loss, w)	# loss对w求导
	w.assign_sub(lr*grads)			# w-=lr*grads
	print(epoch, w.numpy(), loss.numpy())
