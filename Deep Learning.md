# Deep Learning
Github：https://github.com/IvenStarry  
学习视频网站：李沐动手学深度学习  
https://www.bilibili.com/video/BV1if4y147hS

## Chapter 1 ：预备知识
### 数据操作
```python
import torch

# 张量表示一个数值组成的数组，这个数组可能有多个维度
x = torch.arange(12)
print(x)

# .shape 张量的形状
print(x.shape)

# .numel() 元素的总数
print(x.numel())

# reshape 改变张量的形状
x = x.reshape(3, 4)
print(x)

# zeros ones tensor 创建全0、全1、其他常量或者从特定分布中随机采样的数字
print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))
print(torch.tensor([[1, 2], [3, 4], [5, 6]]))

# + - * / ** 常见的标准算术运算符
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x /y, x ** y)

# exp 更多运算 
print(torch.exp(x))

# cat 多个张量连接在一起
x = torch.arange(12, dtype=torch.float32).reshape((3, 4,))
y = torch.tensor([[2.0, 1, 4, 3],
                [1, 2, 3, 4],
                [4, 3, 2, 1]])
print(torch.cat((x, y), dim=0))
print(torch.cat((x, y), dim=1))

# 通过逻辑运算符构建二元张量
print(x == y)

# 对张量中所有元素进行求和会产生一个只有一个元素的张量
print(x.sum())

# 广播机制 如果算术运算发现两个张量形状不同但维度相同，将触发广播机制 这里将a复制列广播成为(3,2) 将b复制行广播为(3,2) 类似NumPy，在大多数情况下，将沿着数组中长度为1的轴进行广播
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a + b)

# 元素访问
print(x[-1]) # 访问最后一行
print(x[1:3]) # 输出第一行到第三行的值（不包含第三行）

# 指定索引将元素写入矩阵
x[1, 2] = 9
print(x)

# 为多个元素赋值相同的值:
x[0:2, :] = 12 # 如果全选列 可以不写后一个 ,:
print(x)

# 运行一些操作可能会导致为新结果分配内存，对于机器学习数百兆的参数更新，不希望总分配没有必要的内存
# id表示Object(对象)在Python中唯一的标识号，内存地址
before = id(y)
y = x + y # 将x和y相加并创建一个新的变量，为结果分配了新的地址，并取名y，之前的内存被析构掉
print(id(y) == before) 

# 执行原地操作，不改变内存地址
z = torch.zeros_like(y) # zeros_like 继承已知张量，根据已知张量的形状，把里面的数据全部置0
print(f"before id(z):{id(z)}")
z[:] = x + y
print(f"after id(z):{id(z)}")

# 如果后续计算中没有重复使用x，也可以使用 x[:] = x + y 或 x += y 减少操作的内存开销
before = id(x)
x += y
print(f"x += y id equal?{id(x) == before}")
x[:] = x + y
print(f"x[:] = x + y id equal?{id(x) == before}")

# 转换为NumPy张量
a = x.numpy()
b = torch.tensor(a)
print(type(a), type(b))

# 将大小为1的张量转换为Python标量(大小不为1无法转换标量)
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
```
### 数据预处理
```python
import os
import numpy as np
# os.markdirs()创建目录 exist_ok设置为True当目录存在时不会报错
os.makedirs(os.path.join('.', 'related_data'), exist_ok=True)
# 创建一个人工数据集，并存储在CSV文件(逗号分隔值)
data_file = os.path.join('.', 'related_data', 'house_tiny.csv')
# ! 为了识别缺失值NaN, NA前面不能有空格
with open(data_file, 'w') as f:
    f.write('NumRoom,Alley,Price\n') # 列明
    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000')

# pandas 从创建的csv文件中加载原始数据集
import pandas as pd
data = pd.read_csv(data_file)
print(data)
print(type(data))

# 处理缺失的数据
# * 方法1：插值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 对于数值列：fillna() 对所有是na的域赋值 mean中的numeric_only设置为True则只处理数值列的均值
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)
# 对于非数值列(类别/离散)：将“NAN”视为一个类别
# pd.get_dummies 将类别变量（categorical variables）转换为独热编码 dummy_na: 是否包括表示缺失值的列
inputs = pd.get_dummies(inputs, dummy_na=True).astype(int)
print(inputs)

import torch
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)

# * 方法2：删除缺失值最多的列
# 实现1
count = 0
count_max = 0
col_names = data.columns.values # .values返回一个ndarray数组
for col_name in col_names:
    count = data[col_name].isna().sum() # isna() 检测缺失值
    print(f"column name:{col_name}, NAN numbers:{count}")
    if count > count_max:
        count_max = count
        drop_col_name = col_name

data_drop = data.drop(drop_col_name, axis=1)
print(data_drop)

# 实现2
NAN_count = data.isna().sum() # 返回每一列的缺失值的总数
print(NAN_count)
label = NAN_count.idxmax() # idxmax返回最大值的索引 argmax返回最大值的下标 max返回最大值
print(label)
new_data = data.drop(label, axis=1)
print(new_data)
```
**Tensor的存储方式：**(参考网站：https://blog.csdn.net/Flag_ing/article/details/109129752)

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409221010269.png)
Tensor A的形状size、步长stride、数据的索引等信息都存储在头信息区，而A所存储的真实数据则存储在存储区；如果我们对A进行截取、转置或修改等操作后赋值给B，则B的数据共享A的存储区，存储区的数据数量没变，共享存储地址，B数据改变A也会因此改变，变化的只是B的头信息区对数据的索引方式

**在Tensor中stride的概念：**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409221014332.png)

**视图与副本的区别：**
- 视图(view)是数据的一个别称或引用，通过该别称或引用亦便可访问、操作原有数据，但原有数据不会产生拷贝。
如果我们对视图进行修改，它会影响到原始数据，物理内存在同一位置，这样避免了重新创建张量的高内存开销，对张量的大部分操作就是视图操作。
- 副本是一个数据的完整的拷贝，如果我们对副本进行修改，它不会影响到原始数据，物理内存不在同一位置

**张量的连续性条件：**$ stride[i] = stride[i + 1] \times size[i] $

**view()与reshape()区别：**
- view()和reshape()均用于调整tensor的形状。
- reshape()不会检查张量的连续性，在不满足连续性条件时不会报错，继续执行出结果，可以使用-1自动推断某些维度。
- 而view()会检查张量的连续性，并在不满足条件时抛出错误，从而提供了更高的安全性，不可以使用-1自动推断某些维度
- reshape方法更强大，可以认为a.reshape = a.view() + a.contiguous().view()。即：在满足tensor连续性条件时，a.reshape返回的结果与a.view()相同，否则返回的结果与a.contiguous().view()相同

> 最新版 PyTorch 2.4.1 view和reshape均可以使用-1自动处理剩下的维度

*示例*
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409221020762.png)
a 和 b 的存储地址一致，数据总量一致，只是对数据的索引不同
```python
# b的结构
import torch
a = torch.arange(9).reshape(3, 3)     # 初始化张量a
b = a.permute(1, 0)  # 对a进行转置
print('struct of b:\n', b)
print('size   of b:', b.size())    # 查看b的shape
print('stride of b:', b.stride())  # 查看b的stride
 
'''
-----运行结果-----
struct of a: (3, 1)
struct of b:
tensor([[0, 3, 6],
        [1, 4, 7],
        [2, 5, 8]])
size   of b: torch.Size([3, 3])
stride of b: (1, 3)   # 注：此时不满足连续性条件
'''

# 因为不满足张量的一致性，直接使用view会报错
a = torch.arange(9).reshape(3, 3)             # 初始化张量a
print(a.view(9))
print('============================================')
b = a.permute(1, 0)  # 转置
print(b.view(9))
 
'''
-----运行结果-----
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
============================================
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
'''

# 直接用view不行，先用contiguous()方法将原始tensor转换为满足连续条件的tensor，在使用view进行shape变换，原理是contiguous()方法开辟了一个新的存储区给b，存储区的物理地址改变，并改变了b原始存储区数据的存放顺序

a = torch.arange(9).reshape(3, 3)      # 初始化张量a
print('storage of a:\n', a.storage())  # 查看a的stride
print('+++++++++++++++++++++++++++++++++++++++++++++++++')
b = a.permute(1, 0).contiguous()       # 转置,并转换为符合连续性条件的tensor
print('size    of b:', b.size())       # 查看b的shape
print('stride  of b:', b.stride())     # 查看b的stride
print('viewd      b:\n', b.view(9))    # 对b进行view操作，并打印结果
print('+++++++++++++++++++++++++++++++++++++++++++++++++')
print('storage of a:\n', a.storage())  # 查看a的存储空间
print('storage of b:\n', b.storage())  # 查看b的存储空间
print('+++++++++++++++++++++++++++++++++++++++++++++++++')
print('ptr of a:\n', a.storage().data_ptr())  # 查看a的存储空间地址
print('ptr of b:\n', b.storage().data_ptr())  # 查看b的存储空间地址

```

### 线性代数
**范数**：可以简单理解为向量的距离
$ \Vert x + y \Vert \leq \Vert x \Vert + \Vert y \Vert $
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409221602725.png)
**各种乘法运算的区别**
|类型|计算方法|示例$ A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$|
|-|-|-|
|*(星乘 asterisk)|矩阵/向量按照对应位置相乘|$ A \ast B = \begin{bmatrix} 1 \cdot 5 & 2 \cdot 6 \\ 3 \cdot 7 & 4 \cdot 8 \end{bmatrix}$|
|·(点乘 dot)|矩阵/向量内积 相同位置的按元素乘积的和|$ A \cdot B = \begin{bmatrix} 1\cdot5 + 2\cdot7 & 1\cdot6 + 2\cdot8 \\ 3\cdot5 + 4\cdot7 &3\cdot6+4\cdot8\end{bmatrix}$|
```python
import torch

# * 标量 由只有一个元素的张量表示
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y, x * y, x / y, x ** y)

# * 向量 可以视为由标量值组成的列表
x = torch.arange(4)
print(x)

# 通过张量的索引来访问任一元素
print(x[3])

# 访问张量的长度
print(len(x))

# 只有一个轴的张量，形状只有一个元素
print(x.shape)

# * 矩阵 指定两个分量m和n来创建一个形状为m×n的矩阵
A = torch.arange(20).reshape(5, 4)
print(A)

# 矩阵的转置
print(A.T)

# 对称矩阵 A=A.T
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B == B.T)

# * 张量 构建具有更多轴(维度)的数据结构
X = torch.arange(24).reshape(2, 3, 4)
print(X)

# 给定具有相同形状的任何两个张量，任何按元素二元运算的结果都将是相同形状的张量
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 分配新内存，将A的一个副本分配给B(深拷贝)
print(A, A + B)

# * 张量算法的基本性质
# 两个矩阵的按元素乘法称为 哈达玛积
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape)

# * 降维
# 计算其元素的和
x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())

# 表示任意形状张量的元素和 调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。 
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A.shape, A.sum())

# 指定求和汇总张量的轴 为了通过求和所有行的元素来降维（轴0），可以在调用函数时指定axis=0。
# 沿着哪个轴求和 相当于把这个轴拍扁 把axis轴的元素去掉，剩下就是求和的结果shape
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1, A_sum_axis1.shape)
print(A.sum(axis=[0, 1])) # 跟sum()一样

# 平均值 一个与求和相关的量
print(A.mean(), A.sum() / A.numel()) # numel() 返回元素总数
print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

# * 非降维求和
# 计算总和或均值时保持轴数不变
sum_A = A.sum(axis=1, keepdim=True)
print(sum_A)

# 通过广播将A除以sum_A
print(A / sum_A)

# 某个轴计算A元素的累加总和
print(A.cumsum(axis=0))
print(A.cumsum(axis=1))

# * 点积dot 相同位置的按元素乘积的和
y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))

# 通过执行按元素乘法，然后求和来表示两个向量的点积
print(torch.sum(x * y))

# * 向量积mv Ax 是一个长度为m的列向量，其 i^th 元素是点积 a^T_i x
print(A.shape, x.shape, torch.mv(A, x))

# * 矩阵乘法mm
B = torch.ones(4, 3)
print(torch.mm(A, B))

# * 范数norm   
# L2范数：向量元素平方和的平凡根  ||x||_2 = sqrt(sum^{n}_{i=1}x_i^2)
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

# L1范数：向量元素的绝对值之和 ||x||_1 = sum^{n}_{i=1}|x_i
print(torch.abs(u).sum())

# 弗罗贝尼乌斯范数 Frobenius norm 矩阵元素的平方和的平方根 ||X||_F = sqrt(sum^{nm}_{i=1}sum^{n}_{j=1}x_{ij}^2)
print(torch.norm(torch.ones((4, 9))))
```
Torch中的一个向量对于计算机来说就是一个数组，没有行向量与列向量之分，如果想要区别行向量和列向量，需要用一个矩阵来表示(torch中一维数组会被视作列向量)
```python
import torch

a = torch.tensor([1, 2 ,3], dtype=torch.float32)
print(a, a.T)
b = torch.ones(3, 3)
print(torch.mv(b, a))
print("一维张量(数组)默认列向量 (3, 3)(3, 1)")
try:
    print(torch.mv(a.T, b))
except RuntimeError as e:
    print("作用在一维张量(数组)上的转置操作不起作用")
```

### 微积分
**梯度**：在等高线上做正交方向，指向值变化最大的方向
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409241009484.png)
**矩阵的导数运算**：
- *分子布局*  
分子为列向量则求导为列向量，即求导结果的维度以分子为主
- *分母布局*  
分母为列向量则求导为列向量，即求导结果的维度以分母为主

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409241045761.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409241107129.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409241040882.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409241052214.png)

### 自动微分
**自动求导的原理**：计算图  
- 将代码分解成操作子
- 将计算表示成一个无环图
- 计算图的构造 
  - 显式构造：先给公式再给值（Tensorflow）
  - 隐式构造：先给值再给公式（PyTorch）

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409241414839.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409241415315.png)
```python
import torch

# * 一个简单的例子
# 假设对 y = 2x.T x 关于列向量x求导
x = torch.arange(4.0)
print(x)

# 计算y对x的梯度前，需要一个地方存储梯度
x.requires_grad_(True) # 等价于 x = torch.arange(4.0, requires_grad=True)
print(x.grad) # Default:None

# 计算y
y = 2 * torch.dot(x, x) # 作内积
print(y)

# 调用反向传播函数来自动计算y关于x每个分量的梯度
y.backward() # ||x||^2 的导数是x.T 所以y对x的导数是4x.T
print(x.grad)

# 默认情况下，PyTorch会累积梯度，需要清除之前的值
x.grad.zero_()
y = x.sum() # y=x1+x2+...+xn 求导全为1.T
y.backward()
print(x.grad)

# * 非标量的反向传播
# 深度学习中，目的不是计算微分矩阵，而是批量中每个样本单独计算的偏导数之和
# 对非标量调用 backward 需要传入一个 gradient 参数，该参数
x.grad.zero_()
y = x * x # 理论上 x是向量 y是向量 求导是一个矩阵，但机器学习绝大部分都是对标量进行求导，因此会在下面加一个sum()函数
y.sum().backward() # 等价于y.backward(torch.ones(len(x)))
print(x.grad)

# * 分离计算
# 将某些计算移动到记录的计算图之外
x.grad.zero_()
y = x * x 
u = y.detach() # detach u将y视作一个常数，而不是与x相关的函数
z = u * x # 所以求导的结果是u

z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

# * Python控制流的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000: # norm()求欧几里得范数 即模长
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
print(a)
d = f(a)
print(d)
d.backward()

print(a.grad == d / a)
```

## Chapter 2 ：线性神经网络
### 线性回归
*显示解*
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409241557141.png)

### 线性回归的从零开始实现
```python
import random
import torch
from d2l import torch as d2l

# * 生成数据集
# 根据带有噪声的线性模型构造一个人造数据集。我们使用线性模型参数w = [2, -3.4]T  b = 4.2和噪声项ε生成数据集及其标签 y = Xw + b + ε
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w))) # X是符合均值为0，方差为1的正态分布的随机数
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print(f'features:{features[0]}, label:{labels[0]}')

# features中的每一行都包含一个二维数据样本，labels中每一个行都包含一个一维标签值(一个标量)
d2l.set_figsize()
d2l.plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), 1) # detach()返回一个新的tensor，从计算图中分离下来的，这样才可以转numpy
d2l.plt.show()

# * 读取数据集
# 定义一个data_iter函数，该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) # 生成样本的索引列表
    random.shuffle(indices) # 打乱索引值，样本是随机读取的，没有特定顺序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, num_examples)] # 拿出当前批次的所有索引值(min的作用防止indices超出总数量)
        )
        yield features[batch_indices], labels[batch_indices] # yield的作用是return的迭代(生成器)，在循环结束前可以在有需要的时候一直返回值
data_iter(64, features, labels)

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, "\n", y)
    break

# * 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# * 定义模型
def linreg(X, w, b):
    return torch.matmul(X ,w) + b

# * 定义损失函数
def squared_loss(y_hat, y): # 均方损失 loss = 1/2n(y_hat-y)^2 样本量n在优化算法中计算
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# * 定义优化算法
def sgd(params, lr, batch_size): # 小批量随机梯度下降
    with torch.no_grad(): # 梯度更新时不要参与梯度计算
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# * 训练过程
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X 和 y 的小批量损失
        l.sum().backward() # l的形状(batch_size, 1)不是标量 使用sum降维
        sgd([w, b], lr, batch_size) # 对梯度进行更新

    with torch.no_grad(): # 评价模型
        train_1 = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_1.mean())}')

# 比较真实参数和通过训练学到的参数来评估训练的成功程度
print(f'w的估计误差:{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差:{true_b - b}')
```

### 线性回归的简洁实现
```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# * 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# * 读取数据集
# 调用框架中现有的API来读取数据
def load_array(data_arrays, batch_size, is_train=True):
    # 构造一个PyTorch数据迭代器
    dataset = data.TensorDataset(*data_arrays) # \ *data_arrays打包做成list传入TensorData
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

# * 定义模型
# 使用框架的预定义好的层
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

# * 初始化模型参数
net[0].weight.data.normal_(0, 0.01) # 查看网络第0层 修改权重参数
net[0].bias.data.fill_(0) # 偏差置0

# * 定义损失函数
# 均方误差 也成为平方L2范数 通过MESLoss()调用
loss = nn.MSELoss()

# * 定义优化算法
# 实例化SGD
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# * 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    
    l = loss(net(features), labels)
    print(f"epoch {epoch + 1}, loss {l:f}") # l:f 打印l 格式为浮点型
```

### Softmax回归
回归与分类的区别:
- 回归估计一个连续值，单连续值输出，自然区间R，跟真实值的区别作为损失
- 分类预测一个离散类别，输出i是预测为第i类的置信度

**损失函数**：
|损失函数|公式|似然函数|
|:-:|:-:|:-:|
|L2 Loss|$l(y,y') = \frac{1}{2}(y - y')^2$|$e^{-l}$|
|L1 Loss|$l(y,y') = \vert y - y' \vert$|$e^{-l}$|
|Huber's Robust Loss|$ l(y,y')= \begin{cases} 1\quad\quad\quad\quad\quad if\quad\vert y-y'\vert > 1  \\ \frac{1}{2}(y-y')^2 \quad otherwise \end{cases}$||

**Softmax回归损失函数梯度的推导**：
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409261506954.png)
**最小二乘法**：  
二乘的意思是样本标签与估计值之差的平方$(y-y')^2$，这个平方就是二乘，最小二乘法的最小就是希望这个平方项最小化，从而预测值与真实值之间的差异足够小，证明模型预测的更加准确
> 不使用 |y-y'| 是因为绝对值函数在R域上不是处处可导，使用平方项处处可导且不影响两者的关系，外面再乘1/2为了使求导后的函数计算更简单

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409261418088.png)

**似然值**：  
真实的情况已经发生，假设有很多模型，在这个概率模型下，发生这种情况(真实情况)的可能性，称为似然值
**最大似然估计**：  
真实值已经发生，本来的概率模型应该是什么样子无法确定，但可以选择似然值最大的模型，这个概率模型和原本的概率模型应该是最接近的，这个方法是最大似然估计法
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409261414232.png)
**对吴恩达逻辑回归问题损失函数的推导**：
P是在某种概率模型下，产生真实事件(图片是否为猫)的似然值，神经网络中用 $W,b$ 来代替概率 $\theta$，这个似然值最大时，认为神经网络训练的模型和人脑中对于判别猫的模型是一致的，因为在训练的时候 $W,b$ 是确定的，不论输入什么图片，输出值也确定，得到的标签要么为1要么为0，无法进行训练，但预测概率 $\hat{y}$ 的结果依赖于 $W,b$ ，因此用神经网络预测概率 $\hat{y}$ 来代替 $W,b$ ，模型概率符合伯努利分布。为了方便计算，将累乘转换为累加的形式，外面乘一个log函数(log不改变单调性)并乘一个负号(习惯求最小值)。
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409261420703.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409261445737.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409261444957.png)

### 图像分类数据集
```python
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display() # 用svg显示图像 清晰度会更高

# MNIST数据集是图像分类中广泛使用的数据集之一，但作为基准数据过于简单，这里使用类似但更复杂的Fashion-MNIST数据集
# * 读取数据集
# 通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存当中
trans = transforms.ToTensor() # 将图像数据从PIL类型变换为32位浮点数格式Tensor
print(type(trans))
mnist_train = torchvision.datasets.FashionMNIST(root="./related_data", train=True,
                                                transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./related_data", train=False,
                                                transform=trans, download=True)

print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape) # 第一维度0表示img图片，1表示标签target，第二维度0表示图片序号 灰度图像channel为1

# 两个可视化数据集的函数
def get_fashion_mnist_labels(labels): # 返回数据集的文本标签
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): # 绘图
    figsize = (num_cols * scale, num_rows * scale) # 画窗大小

    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize) # fig 画窗 ax 坐标系
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # enumerate() 函数用于在迭代过程中同时获取元素的索引和值，返回一个枚举对象，包含了索引和对应的元素
        # 下划线_ 表示不需要的值，减少内存消耗
        if torch.is_tensor(img): # 若数据类型为Tensor
            ax.imshow(img.numpy())
        else: # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
    
    return axes

# 几个样本的图像及其相应的标签
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# * 读取小批量
# 读取一小批量数据，大小为batch_size
batch_size = 256

def get_dataloader_workers(): # 使用4个进程来读取数据
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec') # {} 中实际上存放的是表达式的值，可以在 {} 进行运算。使用content:format 的方式来设置字符串格式

# * 整合所有组件
# 定义 load_data_fashion_mnist函数
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]

    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans) # compose传入list列表

    mnist_train = torchvision.datasets.FashionMNIST(root="./related_data", train=True,
                                                transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./related_data", train=False,
                                                transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

# 指定resize测试
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

**enumerate()函数**
```python
# 枚举列表
fruits = ['apple', 'banana', 'orange']
for index, fruit in enumerate(fruits):
    print(index, fruit)

# zip将多个可迭代对象组合
fruits = ['apple', 'banana', 'orange']
prices = [1.0, 0.5, 0.8]
for index, (fruit, price) in enumerate(zip(fruits, prices)):
    print(index, fruit, price)

# print(list(zip(fruits, prices)))


# 枚举字典的键值对
fruits = {'apple':0.5, 'banana':1.0, 'orange':0.8}
for index, (fruit, price) in enumerate(fruits.items()):
    print(index, fruit, price)
```

**迭代器iter及生成器介绍**

**可迭代对象**：list、tuple、dict、set、str

**迭代器**iterator：
- Iterator对象表示的是一个数据流，Iterator 对象可以被 next() 函数调用并不断返回下一个数据，直到没有数据时抛出 StopIteration 错误。可以把这个数据流看做是一个有序序列，但我们却不能提前知道序列的长度，只能不断通过 next() 函数实现按需计算下一个数据，所以 Iterator 的计算是惰性的，只有在需要返回下一个数据时它才会计算。
- Iterator 甚至可以表示一个无限大的数据流，例如全体自然数。而使用 list 是永远不可能存储全体自然数的。

**生成器**：  
使用yield语句来生成迭代器，一种返回一个值的迭代器，每次从该迭代器取下一个值，可以节省内存空间和计算资源

**生成器表达式**：
生成器表达式是用圆括号来创建生成器，其语法与推导式相同，只是将 [] 换成了 () 。 生成器表达式会产生一个新的生成器对象。

**迭代器和生成器的区别**：
- 迭代器是实现了迭代器协议（即__iter__()和__next__()方法）的对象。生成器则是使用了yield关键字的函数，当这个函数被调用时，它返回一个生成器对象。
- 生成器在每次产生一个值后会自动保存当前的状态，下次调用时会从上次离开的地方继续执行。而迭代器则不会自动保存状态，它依赖于用户显式地请求下一个值。
- 生成器在迭代过程中只会生成当前需要的值，而不是一次性生成所有的值，所以它可以处理大数据集，而不会耗尽内存。而迭代器则可能需要一次性生成所有的值。
- 迭代器可以被多次迭代，每次迭代都会从头开始。而生成器只能被迭代一次，因为它不保存整个序列，只保存当前的状态。
- 生成器更加灵活，可以使用任何种类的控制流语句（如if，while等）。而迭代器则需要在__next__()方法中实现所有的控制流逻辑。
- 总的来说，生成器是一种特殊的迭代器，它更加简洁，易于理解，同时也更加强大和灵活

```python
# 迭代器与列表
''' 
区别: 
- 迭代器经历一次for in后，再次遍历返回空
- 列表遍历多少次，表头位置是第一个元素
- 迭代器遍历后，指向最后元素的下一个位置
'''
list_test = [1, 2, 3]
b = iter(list_test)
for i in b:
    print(i)

print("--------第二次遍历开始----------")
for i in b:
    print(i)
print("--------第二次遍历结束----------")

# 迭代器与字典
# next是一个内建函数，用于从迭代器中获取下一个项目
dict_test = {'Iven':20, 'Rosenn':22}
print(next(iter(dict_test.keys())))
print(next(iter(dict_test.values())))

iter_dict_test = iter(dict_test)
print(next(iter_dict_test)) # 字典迭代器默认遍历字典的键key
print(next(iter_dict_test))
try:
    print(next(iter_dict_test))
except StopIteration as r:
    print(f"已到迭代器尾部，停止迭代")


# 生成器
# 使用圆括号创建生成器
print(type([i for i in range(5)]))# 推导式表达式 for 迭代变量 in 可迭代对象 [if 条件表达式]
print(type((i for i in range(5))))

a = (i for i in range(5))
print(next(a))
print(next(a))

a = [ i + 1 for i in range(5)] # 列表推导式
print(a)

# 生成器函数 yield
# 函数如果包含 yield 指令，该函数调用的返回值是一个生成器对象，此时函数体中的代码并不会执行，只有显式或隐式地调用 next 的时候才会真正执行里面的代码。yield可以暂停一个函数并返回此时的中间结果。该函数将保存执行环境并在下一次恢复。
def fun():
    for i in range(5):
        yield i

f = fun()
print(next(f)) # 显式
print(next(f))

for i in fun(): # 隐式
    print(i)
```
### Softmax回归的从零开始实现


### Softmax回归的简洁实现