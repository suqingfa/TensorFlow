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

# 2.8 鸢尾花分类实现
- 安装依赖  
    conda install scikit-learn

  [实现](2_iris.py)

# 3.神经网络优化

# 4.神经网络八股

# 5.网络八股扩展

# 6.卷积神经网络

# 7.循环神经网络
