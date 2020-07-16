# 1.环境安装 
- 安装 Miniconda  
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe
- 运行 Anaconda Prompt (Miniconda3)

- 设置conda代理  
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/  
	conda config --set show_channel_urls yes

- 设置pip代理  
	pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

- 安装tensorflow  
	conda install tensorflow-gpu

- 更新  
	conda update --all  

- [测试环境](1_env.py)

# 2.神经网络计算过程

## 2.1 当前人工智能主流方向-连接主义
仿生学，模仿神经元连接关系  
MP模型  
![MP模型](md_img/2.1mp.png)  
为了求解简单，去掉非线性函数得到  
Y = W * X + b

## 2.2 前向传播
给定X计算出Y

## 2.3 损失函数
预测值y与标准值y_之间的差距  
损失函数可以定量判断W、b的优劣，
当损失函数值最小时，参数W、b会出现最优值  
损失函数定义有多种方法 均方误差MSE

## 2.4 梯度下降
目的：寻找一组参数W、b使得误差函数最小  
梯度：函数对各参数求偏导后的向量 函数梯度下降方向就是函数减小方向
梯度下降法：沿损失函数梯度下降方向，寻找损失函数最小值，得到最优参数的方法  
计算公式  
![w](https://latex.codecogs.com/png.download?w_%7Bt+1%7D%3Dw_%7Bt%7D-lr*%5Cfrac%7B%5Cpartial%20loss%7D%7B%5Cpartial%20w_%7Bt%7D%7D)

![b](https://latex.codecogs.com/png.download?b_%7Bt+1%7D%3Db_%7Bt%7D-lr*%5Cfrac%7B%5Cpartial%20loss%7D%7B%5Cpartial%20b_%7Bt%7D%7D)

![y](https://latex.codecogs.com/png.download?w_%7Bt+1%7D*x+b_%7Bt+1%7D%5Crightarrow%20y)

## 2.5 学习率
lr设置过小时，参数更新过程缓慢 
lr设置过大时，梯度可能在最优值左右来回震荡

## 2.6 反向传播更新参数
从后向前，逐层求损失函数对每层参数的偏导数，迭代更新所有参数

[示例](2_back_propagation.py)

## 2.7 常用函数 
- 张量定义  
tf.constant(
	value, dtype=None, shape=None, name='Const'
)

- 转换为张量  
tf.convert_to_tensor(
	value, dtype=None, dtype_hint=None, name=None
)

- 填充张量  
tf.zeros(
	shape, dtype=tf.dtypes.float32, name=None
)  
tf.ones(
	shape, dtype=tf.dtypes.float32, name=None
)  
tf.fill(
	dims, value, name=None
)

- 正态分布随机张量  
tf.random.normal(
	shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None
)

- 截断式正态分布随机张量  
tf.random.truncated_normal(
	shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None
)

- 均匀分布随机张量  
tf.random.uniform(
	shape, minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
)

- 乱序  
tf.random.shuffle(
    value, seed=None, name=None
)

- 类型转换  
tf.cast(
	x, dtype, name=None
)

- 数学运算  
    - 算术运算，对张量对应元素进行算术运算   
    add\subtract\multiply\divide\square\pow\sqrt...    
    tf.math.ag(x, y, name = None)

    - 压缩运算，对张量指定方向进行压缩运算 
    axis 压缩维度
    any\max\mean\min\prob\std\sum\variance  
    tf.math.reduce_ag(
    input_tensor, axis=None, keepdims=False, name=None
    )

- 归一化指数函数  
tf.math.softmax(
    logits, axis=None, name=None
)

- 独热码  
tf.one_hot(
    indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None
)

- 将变量标记为可训练  
tf.Variable(
    initial_value=None, trainable=None, validate_shape=True, caching_device=None,
    name=None, variable_def=None, dtype=None, import_scope=None, constraint=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.compat.v1.VariableAggregation.NONE, shape=None
)

- 矩阵乘法  
tf.matmul(
    a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False,
    a_is_sparse=False, b_is_sparse=False, name=None
)

- 输入特征、标签配对  
tf.data.Dataset.from_tensor_slices(
    tensors
)

## 2.8 鸢尾花分类实现
- 安装依赖  
    conda install scikit-learn

  [实现](2_iris.py)

# 3.神经网络优化

## 3.1 预备知识
- 条件函数 条件成立时返回x中的元素，否则返回y中的元素    
tf.where(
    condition, x=None, y=None, name=None
)  

- 随机数 [0, 1)区间的随机数  
numpy.random.RandomState.rand(d0, d1, ..., dn)

- 数组叠加  
numpy.vstack(tup)

## 3.2 神经网络复杂度
- 空间复杂度
    - 层数 隐藏层数+输出层
    - 参数个数 总w + 总b

- 时间复杂度
    - 乘加运算次数

## 3.3 指数衰减学习率
    lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)

## 3.4 激活函数
对于神经网络中的每个计算单元有 Y = f(X*W + b)  
函数f称为激活函数 通过激活函数，可以计算非线性问题

- Sigmoid激活函数  
tf.nn.sigmoid(
    x, name=None
)  
表达式 f(x) = 1 / ( 1 + e^x)  
特点
    - 易造成梯度消失
    - 输出非0均值，收敛慢
    - 幂运算，训练时间长

- Tanh激活函数  
tf.nn.tanh(
    x, name=None
)  
表达式　f(x) = (1 - e^(-2x)) / (1 + e^(-2x))  
特点
    - 均值为0
    - 易造成梯度消失
    - 幂运算，训练时间长

- Relu激活函数  
tf.nn.relu(
    features, name=None
)  
表达式 f(x) = max(x, 0)  
优点
    - 解决梯度消失问题(在正区间)
    - 计算速度快
    - 收敛速度快
缺点
    - 输出均值非0，收敛慢
    - Dead ReLU问题 某些神经元可以永远无法激活，导致参数不会更新
 
- Leaky Relu函数  
tf.nn.leaky_relu(
    features, alpha=0.2, name=None
)  
表达式 f(x) = max(x, ax)  
理论上讲LeakyRelu函数有Relu函数的所有优点，但不存在DeadReLU问题
但实际操作中，并没有完全证明

## 3.5 损失函数 
预测值y与标准值y_的差距  [示例](3_loss.py)

- 均方损失函数  
loss_mse = tf.math.reduce_mean(tf.math.squary(y - y_))

- 自定义损失函数  

- ce(Cross Entropy)交叉熵损失函数  
tf.losses.categorical_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0
)  

- softmax与交叉熵结合  
tf.nn.softmax_cross_entropy_with_logits(
    labels, logits, axis=-1, name=None
)

## 3.6 欠拟合和过拟合
欠拟合 对训练集学习不够彻底  
缓解方法  
- 增加输入特征项
- 增加网络参数
- 正则化减小参数

过拟合 模型对数据拟合的太好，但对从未见过的新数据确难以做出正确分类  
缓解方法
- 数据清洗
- 增大训练集
- 采用正则化
- 增大正则化参数
- 

## 3.7 正则化减小过拟合
[示例](3_regularizer.py)  
正则化缓解过拟合
loss = loss(y, y_) + REGULARIZER * l_loss(w)  
![正则化](md_img/3_regularizer.jpg)
如果不加L1和L2正则化的时候，对于线性回归这种目标函数凸函数的话，我们最终的结果就是最里边的紫色的小圈圈等高线上的点。
当加入L1正则化的时候，我们先画出 F = |w1| + |w2| 的图像，也就是一个菱形，代表这些曲线上的点算出来的都为F。那我们现在的目标是不仅是原曲线算得值要小（越来越接近中心的紫色圈圈），还要使得这个菱形越小越好（F越小越好）。

- L1正则化 大概率使得很多参数变为零，可以减少参数数量，降低复杂度  

- L2正则化 使得参数接近零但不为零，可以降低复杂度  
tf.nn.l2_loss(
    t, name=None
)

## 3.8 神经网络参数优化器
等优化参数w，损失函数loss，学习率lr，每次迭代一个batch，当前总迭代次数t

参数优化过程:
- 计算t时刻损失函数对于当前参数的梯度  ![g_t](https://latex.codecogs.com/png.latex?g_t%20%3D%20%5Cfrac%7B%5Cpartial%20loss%7D%7B%5Cpartial%20w_t%7D)
- 计算t时刻的一阶运量 m_t 和二阶运量 V_t
- 计算t时刻下降梯度 n_t = lr * m_t / sqrt(V_t)
- 计算t+1时刻参数 ![w_{t+1}](https://latex.codecogs.com/png.latex?w_%7Bt&plus;1%7D%20%3D%20w_t%20-%20%5Ceta%20_t%20%3D%20w_t%20-%20lr%20*%20m_t%20/%20%5Csqrt%7BV_T%7D)

一阶运量 与梯度相关的函数 
二阶运量 与梯度平方相关的函数

常用优化器:
- SGD 
m_t=g_t V_t=1  
tf.optimizers.SGD(
    learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD', **kwargs
)

- SGDM
m_t=p*m_{t-1}+(1-p)*g_t V_t=1  

- Adagrad
m_t=g_t ![V_t](https://latex.codecogs.com/png.latex?V_T%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bt%7Dg_%7Bi%7D%5E%7B2%7D)  
tf.optimizers.Adagrad(
    learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,
    name='Adagrad', **kwargs
)

- RMSprop  
m_t=g_t V_t=p*V_{t-1}+(1-p)*(g_t)^2
tf.optimizers.RMSprop(
    learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
    name='RMSprop', **kwargs
)

- Adam  
tf.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam', **kwargs
)

# 4.神经网络八股

# 5.网络八股扩展

# 6.卷积神经网络

# 7.循环神经网络
