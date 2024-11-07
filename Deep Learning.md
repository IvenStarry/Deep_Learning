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
|*(星乘 asterisk)哈达玛积|矩阵/向量按照对应位置相乘|$ A \ast B = \begin{bmatrix} 1 \cdot 5 & 2 \cdot 6 \\ 3 \cdot 7 & 4 \cdot 8 \end{bmatrix}$|
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

# * 点积dot 相同位置的按元素乘积的和(先求哈达玛积，再求和)
y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))

# 通过执行按元素(哈达玛积)乘法，然后求和来表示两个向量的点积
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
求导结果的行数和分子相同
- *分母布局*  
求导结果的行数和分母相同

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101532632.jpg)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101533623.jpg)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101533284.jpg)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101533835.jpg)
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
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202409271608029.png)
```python
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# * 初始化模型参数
# 对于Softmax回归，输入是一个向量，因此将展平每个图像，将它们视为长度为784的向量
num_inputs = 784
num_outputs = 10 # 10个类别

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# * 定义Softmax操作
# 给定一个矩阵X，我们可以对所有元素求和
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

# 实现Softmax
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True) # 对行进行求和(每个样本是一行，上面将图像展平)
    # print(partition)

    return X_exp / partition # 利用广播机制

# 将没每个元素变成一个非负数，根据概率原理，每行总和为1
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))

# * 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b) # W.shape[0]=784 -1这里计算的是batch_size * dims

# * 定义损失函数
# 创建数据y_hat，其中包含2个样本在3个类别的预测概率，使用y作为y_hat中概率的索引
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y]) # 索引数组 第一个子数组选择行 第二个子数组选择列[[0, 1], [0, 2]] 输出[0,0]和[1,2]的数据

# 实现交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y]) # len计算第0维度
print(cross_entropy(y_hat, y))

# 将预测类别与真实y元素进行比较
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # 判断张量是否大于大于1维度，判断列(第二个维度是否大于1)
        y_hat = y_hat.argmax(axis=1)
    # 由于==对数据类型比较敏感，为了确保y_hat y的数据类型一致，将y_hat转为y的数据类型，结果是一个包含0(错)与1(对)的张量
    cmp = y_hat.type(y.dtype) == y 

    return float(cmp.type(y.dtype).sum()) # 将cmp类型转为tensor并求和，得到正确预测的数量
# 计算正确率
print(accuracy(y_hat, y) / len(y))

# * 分类精度
# 评估在任意模型net的准确率
def evaluate_accuracy(net, data_iter):
    '''
    isinstance()用来判断一个对象是否是一个已知的类型 
    isinstance(object,classtype)
    object -- 实例对象。
    classtype -- 可以是直接或间接类名、基本类型或者由它们组成的元组。
    '''
    if isinstance(net, torch.nn.Module):
        net.eval() # 转评估模型
    
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    
    return metric[0] / metric[1]

# 示例创建2个变量，用于存储正确预测和预测的总数量
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx): # 使对象可以使用索引操作
        return self.data[idx]

'''
CPU性能不够，可能会导致发生DataLoader worker exited unexpectedly

原因：采用多进程加载数据时，python需要重新导入模块以启动每个子进程，这个重复过程会执行模块中的顶级代码，若CPU性能不足以多次调用模块，则会报错

解决方法：
1. 在d2l包中的torch模块get_dataloader_workers()返回值改为0 不启用多线程 这里采用方法1
2. 将下面代码放入if __name__ == '__main__'，可以防止模块被多次导入时重复执行evaluate_accuracy函数

'''
print(evaluate_accuracy(net, test_iter))

# * 训练
# Softmax回归的训练
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    
    metric = Accumulator(3)

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer): # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            # 这里乘len(y)的原因是 pytorch自动对loss取均值
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size.numel()) # loss累加值 正确个数 样本总个数
        
        else: # 使用自己写的优化器和损失函数
            l.sum().backward() # 自己写的损失是一个向量
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    
    return metric[0] / metric[2], metric[1] / metric[2] # 返回训练loss和正确率acc

# 定义一个在动画中绘制数据的实用程序类 (暂时跳过)
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        d2l.plt.pause(0.01)
        display.clear_output(wait=True)

# 训练函数
def train_ch3(net,train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater) # 训练损失
        test_acc = evaluate_accuracy(net, train_iter) # 测试集精度
        animator.add(epoch + 1, train_metrics + (test_acc, ))
    
    train_loss, train_acc = train_metrics

lr = 0.1
# 使用小批量随机梯度下降来优化模型的损失函数
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

# 训练模型10个迭代周期
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
d2l.plt.show()

# * 预测
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break

    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()

predict_ch3(net, test_iter)
```

### Softmax回归的简洁实现
```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# * 初始化模型参数
# Softmax回归的输出层是一个全连接层
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m): # m是Layer层
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01) # 初始化

net.apply(init_weights)

# * 重新审视Softmax的实现
# 在交叉熵损失函数中传递未归一化的预测，并同时计算Softmax及其对数
loss = nn.CrossEntropyLoss(reduction='none') 

# * 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# * 训练
# 调用之前定义的训练函数来训练模型
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()
```

## Chapter 3 : 多层感知机
### 多层感知机
**感知机**：二分类问题，不能拟合XOR（异或）函数，只能线性分割面
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410080915781.png)
**收敛定理**：存在余量 $\rho$ 保证感知机可以找到分割线，值越大，代表两个类别越容易分割开来
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410080919489.png)
**多层感知机**：使用隐藏层和激活函数来得到非线性模型，用多个分割线来完成分类任务，可以完成XOR问题
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410080947469.png)
多层感知机与Softmax回归的区别：多层感知机多了隐藏层  
**激活函数**：
|激活函数名|公式|作用|
|-|-|-|
|Sigmoid|$sigmoid(x)=\frac{1}{1+\exp{-x}} $|将输入变换为区间(0, 1)上的输出|
|Tanh|$ tanh(x)=\frac{1-\exp{-2x}}{1+\exp{-2x}} $|将输入变换为区间(-1, 1)上的输出|
|ReLU|$ ReLU(x) = max(x,0) $|仅保留正元素并丢弃所有负元素|

### 多层感知机的从零开始实现
```python
import torch
from d2l import torch as d2l
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# * 初始化模型参数
# 实现一个具有单隐藏层的多层感知机，它包含256个隐藏单元
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# * 激活函数
# 实现ReLU
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a) 

# * 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1) # @ 是矩阵乘法
    return (H @ W2 + b2)

# * 损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# * 训练
# 多层感知机训练过程与softmax回归的训练过程相同
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

### 多层感知机的简洁实现
```python
import torch
from torch import nn
from d2l import torch as d2l

# * 模型
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

### 模型选择、欠拟合和过拟合
**训练误差和泛化误差**：
- 训练误差：模型在训练数据上的误差
- 泛化误差：模型在新数据上的误差

**训练数据集、验证数据集和测试数据集**：
- 训练数据集：训练模型参数
- 验证数据集：选择模型超参数，用于评估模型好坏的数据集
- 测试数据集：只用一次的数据集

**K-则交叉验证**：在没有足够多数据时使用(非大数据集)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410081132549.png)

**过拟合和欠拟合**：
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410081222437.png)

**模型容量**：
- 拟合各种函数的能力
- 低容量模型难以拟合训练数据
- 高容量模型可以记住所有的训练数据

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410081223930.png)

**VC维**:等于一个最大的数据集大小，不管如何给定标号，都存在一个模型对它进行完美分类，VC维可以衡量训练误差和泛化误差之间的间隔
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410081525947.png)

**数据复杂度**：
- 样本个数
- 每个样本的元素个数
- 时间、空间结构
- 多样性

总结：模型容量需要匹配数据复杂度，否则会导致欠拟合或过拟合
```python
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# * 多项式回归
# * 生成数据集
max_degree = 20 # 多项式最大阶数
n_train, n_test = 100, 100 # 训练和测试集大小
true_w = np.zeros(max_degree) # 分配空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) # 只有前四个有数值，后面多项式的系数是噪音

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1) # gamma(n) = (n-1)!

labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# 看一下前两个样本
true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
# print(features[:2], poly_features[:2, :], labels[:2])

# * 对模型进行训练和测试
# 定义函数评估损失
def evaluate_loss(net, data_iter, loss):
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    
    return metric[0] / metric[1]

# 定义训练函数
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')

    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))

    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                                batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                    evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

# * 三阶多项式函数拟合(正常)
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:]) # 从多项式特征中选择前四个维度

# * 线性函数拟合(欠拟合)
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:]) # 从多项式特征中选择前两个维度

# * 高阶多项式函数拟合(过拟合)
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:]) # 从多项式特征中选择全部维度
```

### 权重衰减
**L2正则化**：
- 硬性限制
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410081637533.png)
- 柔性限制
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410081639202.png)

**参数更新法则**：
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410081650439.png)

```python
import torch
from torch import nn
from d2l import torch as d2l

# ? 高维线性回归
# * 从零开始实现
# 生成数据集
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 定义训练代码实现
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                    d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())
    d2l.plt.waitforbuttonpress()

# # 忽略正则化直接训练
# train(lambd=0)

# # 使用权重衰减
# train(lambd=3)

# * 简洁实现
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd}, # 在这里设置超参数
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                        (d2l.evaluate_loss(net, train_iter, loss),
                        d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
    d2l.plt.waitforbuttonpress()

train_concise(0)
train_concise(3)
```

### 暂退法(Dropout)
丢弃发将一些输出项随机置0来控制模型复杂度，常作用在多层感知机的隐藏层输出上，丢弃概率是超参数
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410082023639.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410082024947.png)
```python
import torch
from torch import nn
from d2l import torch as d2l

# * 从零开始实现
def dropout_layer(X, dropout):
    '''
    assert, 断言语句，可以看做是功能缩小版的if语句，它用于判断某个表达式的值
    如果值为真，则程序可以继续往下执行
    反之，Python 解释器会报 AssertionError 错误。
    '''
    assert 0 <= dropout <= 1

    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    
    mask = (torch.rand(X.shape) > dropout).float() # rand生成[0,1)的随机数与失活率作比较
    return mask * X / (1.0 - dropout)

# 测试
X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))

# 定义模型
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1,
                num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# 训练和测试
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction = 'none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.waitforbuttonpress()

# * 简洁实现
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(dropout1), 
    nn.Linear(256, 256), 
    nn.ReLU(), 
    nn.Dropout(dropout2),
    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.waitforbuttonpress()
```

### 数值稳定性和模型初始化
**神经网络的梯度**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410191451120.png)
**数值稳定性的两个问题**：
- **梯度爆炸**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101539328.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101534041.jpg)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101540466.png)
如果 d (神经网络层数) - t (第几层)较大且w矩阵中会有一些大于1的权值 ，则最终结果累乘起来会很大
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101545453.png)
- **梯度消失**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101550796.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101551778.png)
如果 d (神经网络层数) - t (第几层)较大且上一层输出值很大导致激活函数梯度过小 ，则最终结果累乘起来会很小
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101554053.png)
- 总结：数值过大或过小均会导致数值问题，尤其是深度网络模型
- 解决办法：
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101556905.png)

**理想的参数**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101601223.png)

**权重初始化**
训练开始时梯度更容易更加大，而在最优解附近梯度一般较小
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101658885.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101700683.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101640268.jpg)
方差与期望的关系
$ D(X)=E(X^2)−E^2(X) $
独立同分布随机变量的乘积的期望，等于各自期望的乘积，此处各自期望均等于0
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101709033.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101709284.PNG)

**Xavier初始化方法**  
因为每一层输出维度不能确定，所以同时实现如图公式很难满足，但可以做权衡尽量满足要求
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101713785.png)

为了使前面每一层输出均值为0，方差相等，则激活函数 $ \sigma(x)=\alpha x + \beta = x$
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101717778.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101719518.png)

为了满足数值稳定性，激活函数在x=0附近应满足$f(x) = x$，由泰勒展开可以看出，$tanh(x)$ 和 $relu(x)$均满足，但sigmoid函数不满足，因此需要调整
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410101721259.png)

**提升数值稳定性的方法**：
- 合理的权重初始值
- 合理的激活函数

### 实战Kaggle比赛：预测房价
**Z-score 数值标准化原理**:
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410111146409.jpg)

**评估模型的指标**：相对误差
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410111458315.jpg)

## Chapter 4 : 深度学习计算
### 模型构造
```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
print(net(X))

# * 自定义块
class MLP(nn.Module):
    def __init__(self):
        super().__init__() # super()调用父类函数
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    
    def forward(self, X):
        return self.out(F.relu(self.hidden(X))) # F中的relu是实现了一个函数，而nn.relu实现的是对象

# 实例化多层感知机的层
net = MLP()
print(net(X))

# * 顺序块
# Sequential类的工作原理
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block # 将输入进来的需要的层放入modules，按序字典，自己作为自己的key
    
    def forward(self, X):
        for block in self._modules.values():
            X =  block(X) # block有次序，将X输入进去会按序得到输出
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))

# * 在前向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
print(net(X))

# 混合搭配各种组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
    
    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP()) # 嵌套块(灵活性)
print(chimera(X))
```

### 参数管理
```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X)) # # 通过call魔术方法实现让类可以像函数一样调用

# * 参数访问
print(net[2].state_dict()) # 通过getitem魔术方法实现索引访问

# 目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad == None) # 反向传播后才会有值

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['2.bias'].data)

# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))

# 查看网络结构
print(rgnet)

# * 参数初始化
# 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01) # _的意思是替换掉原有参数(原地操作)，而不是返回一个值
        nn.init.zeros_(m.bias)
net.apply(init_normal) # apply 对net里的所有层进行循环(for loop)，将module传入进init_normal进行调用
print(net[0].weight.data[0], net[0].bias.data[0])

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])

# 对某些块应用不同的初始化方法
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight) # 均匀分布

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

# * 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print(
            'Init',
            *[(name, param.shape) for name, param in m.named_parameters()][0]
        )
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5 # 保留绝对值大于5的权重(若权重大于5 判断为True即1 保留，否则为0 舍去)

net.apply(my_init)
print(net[0].weight[:2])

# 或者直接设置参数
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])

# * 参数绑定
# 共享权重
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
print(net(X))
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100 # shared指向同一片内存
print(net[2].weight.data[0] == net[4].weight.data[0])
```

### 自定义层
```python
import torch
import torch.nn.functional as F
from torch import nn

# * 构造一个不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

# 将层作为组件合并到更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean()) # 检查均值是否为0

# * 带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units): # 输入数和输出数
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

dense = MyLinear(5, 3)
print(dense.weight)

# 自定义层执行前向传播
print(dense(torch.rand(2, 5)))

# 自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
```

### 读写文件
```python
import torch
from torch import nn
from torch.nn import functional as F

# * 加载和保存张量
# 单个张量
x = torch.arange(4)
torch.save(x, 'related_data/x-file')

x2 = torch.load('related_data/x-file', weights_only=True)
print(x2)

# 存储一个张量列表，然后把它们读回内存
y = torch.zeros(4)
torch.save([x, y], 'related_data/x-files')
x2, y2 = torch.load('related_data/x-files', weights_only=True)
print((x2, y2))

# 写入或读取从字符串映射到张量的字典
mydict = {'x':x, 'y':y}
torch.save(mydict, 'related_data/mydict')
mydict2 = torch.load('related_data/mydict', weights_only=True)
print(mydict2)

# * 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 存储模型参数
torch.save(net.state_dict(), 'related_data/mlp.params')

# * 恢复模型
# 需要先实例化多层感知机模型，再拂去参数
clone = MLP()
clone.load_state_dict(torch.load('related_data/mlp.params', weights_only=True))
print(clone.eval())

Y_clone = clone(X)
print(Y_clone == Y)
```

### GPU
```python
import torch
from torch import nn

# * 计算设备
print(torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1'))

# 查询GPU数量
print(torch.cuda.device_count())

# 请求在GPU不存在的情况下使用CPU运行代码
def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus(): # 返回所有可用的GPU
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())

# * 张量与GPU
# 查询张量所在的设备
x = torch.tensor([1, 2, 3])
print(x.device)
# 存储在GPU上
x = torch.ones(2, 3, device=try_gpu())
print(x)
# 在第二个GPU上创建一个随机张量
y = torch.rand(2, 3, device=try_gpu(1))
print(y)

# 复制 
# 计算x+y，必须要让x和y位于同一个GPU上
z = x.cuda(1)
print(x)
print(z)

# 现在数据在同一个GPU上
print(y + z)
print(z.cuda(1) is z) # 变量Z已经存在于第二个GPU上,调用Z.cuda(1)它将返回Z，而不会复制并分配新内存。

# * 神经网络与GPU
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(x))

# 确认模型参数存储在同一个GPU上
print(net[0].weight.data.device)
```

## Chapter 5 : 卷积神经网络
### 从全连接层到卷积
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151019025.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151022337.PNG)
**权值共享**：对于输入同一张输入图像，用同一个卷积核去提取图像的特征  
**平移不变性**：对于同一张图及其平移后的图像，都能输出相同的结果(不管检测对象位于图像中那个位置，神经网络的前面几层对相同的图像区域有相似的反应)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151016039.png)
**空间局限性**：神经网络的前面几层只探索输入图像的局部区域，而不过度在意图像中相隔较远区域的关系
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151017157.png)
对全连接层使用平移不变性和局部性得到卷积层
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151017888.png)

### 图像卷积
**交叉相关**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151024815.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151025769.png)
**交叉相关&卷积的联系**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151025922.png)
g(m, n)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151032734.png)
卷积核
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151033330.png)
```python
import torch
from torch import nn
from d2l import torch as d2l

# * 互相关运算
def corr2d(X, K): # X输入 K卷积核
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) # 卷积结果大小
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

# 验证二维互相关运算的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))

# * 卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# * 图像中目标的边缘检测
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
K = torch.tensor([[1.0, -1.0]])

# 检测结果
Y = corr2d(X, K)
print(Y)

# 卷积核K只可以检测垂直边缘
print(corr2d(X.T, K)) # .t()转置 跟.T 一样

# * 学习卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y ) ** 2 # 损失
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad # 梯度下降
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {l.sum():.3f}')

# 学习的权重张量
print(conv2d.weight.data.reshape((1, 2)))
```

### 填充和步幅
填充
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151634621.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151636571.png)
如果卷积核尺寸为偶数，上侧填充多一行(向上取整)，下侧填充少一行(向下取整)

步幅
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151642641.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151644455.png)
```python
import torch
from torch import nn

# * 填充
def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape) # 元组的连接
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:]) # 去掉前两维 只看宽高

# 填充固定高度宽度
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1) # 上下左右各填充1行或1列
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape) # 8-3+1*2+1

# 填充不同的高度和宽度
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape) # (8-5+2*2+1, 8-3+1*2+1)

# * 步幅
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape) # (8-3+1*2+2)/2 向下取整

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape) # ((8-3+0*2+3)/3, (8-5+1*2+4)/4) 向下取整
```

### 多输入多输出通道
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151718305.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151721888.png)
> 每个卷积核是i维，总共有o个卷积核，因此输出y的通道数是o
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151731123.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410151733915.png)
```python
import torch
from d2l import torch as d2l

# * 多输入通道
def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K)) # 绑定输入x和卷积核

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K)) 

# * 多输出通道
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0) # stack 沿着轴0进行拼接

K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
print(corr2d_multi_in_out(X, K))

# * 1 × 1卷积层
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X) # 相当于全连接层的矩阵乘法
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
```

### 池化层
如果有像素1移动位置，则边缘会输出0，卷积输出太敏感会导致检测效果下降
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410152054609.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410152059861.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410152101918.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410152102975.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410152103184.png)
```python
import torch
from torch import nn
from d2l import torch as d2l

# * 最大池化层和平均池化层
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] =  X[i:i + p_h, j:j +p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

# 验证
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'avg'))

# * 填充和步幅
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
pool2d = nn.MaxPool2d(3) # torch框架中步幅与池化窗口大小一致 (4-3+0+3)/3

# 手动设定填充和步幅
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X)) # (4-3+1*2+2)/2

# 设定任意大小池化窗口
pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
print(pool2d(X)) # ((4-2+1*2+2)/2, (4-3+1*2+3)/3)

# * 多个通道
X = torch.cat((X, X + 1), 1) # 在channel即第一维度上进行叠加
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
```

### 卷积神经网络(LeNet)
用于手写数字识别
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410152136847.png)
```python
import torch 
from torch import nn
from d2l import torch as d2l

# * LeNet
class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), # 原数据集是(32, 32)
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

# 检查模型
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape) # 显示层名，显示层输出形状

# * 模型训练
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 对evaluate_accuracy进行修改 在gpu上进行计算
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            # 如果没有指定device 那么把参数的第一个元素取出来看它在哪一个device上
            print()
            device = next(iter(net.parameters())).device # parameters是一个生成器
        
        metric = d2l.Accumulator(2)

        for X, y in data_iter:
            if isinstance(X, list): # 是list就每一个挪一下
                X = [x.to(device) for x in X]
            else: # 是tensor 一次性全部挪
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
        
        return metric[0] / metric[1]

#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                            (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
            f'on {str(device)}')

# * 训练和评估LeMet-5模型
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Chapter 6 : 现代卷积神经网络
### 深度卷积神经网络(AlexNet)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171500171.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171503336.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171504118.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171506751.png)
数据增强：增加随机光照条件，减少卷积神经网络对于光照的敏感度
```python
import torch
from torch import nn
from d2l import torch as d2l

# * 模型设计
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

# 查看形状
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape:\t', X.shape)

# * 读取数据集
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# * 训练
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

### 使用块的网络(VGG)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171547816.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171548985.png)
总结
- 使用可重复使用的卷积块来构建深度卷积神经网络
- 不同的卷积块个数和超参数可以得到不同复杂度的变种

```python
import torch
from d2l import torch as d2l
from torch import nn

# * VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []

    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# * VGG网络
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) # 定义模型卷积层个数和输出的通道数
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    return nn.Sequential(
        *conv_blks, 
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), # 池化5次 224 -> 7
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )
net = vgg(conv_arch)

# 观察输出形状
X = torch.randn((1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'Output shape:\t', X.shape)

# * 训练
# 因为vgg-11比alexnet计算量更大，因此构建一个通道数少的网络
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

# 正式训练
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

### 网络中的网络(NiN)
全连接层所需要的参数太多，占内存以前硬件跟不上，NiN可以减少参数
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171642601.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171642901.png)
输入通道是1000，对每一个channel做池化，每一个层拿出一个值，做该类别的预测
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171644107.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171650382.png)
总结：
- 全连接层占内存
- 卷积核大占内存
- 层数越多占内存
- 模型所占内存越大越难以训练
- 图像尺寸减半，通道数指数增长，可以很好地保留特征
- 综上，应多用1*1，3*3卷积核，AdaptiveAvgPool2d代替FC层

```python
import torch
from torch import nn
from d2l import torch as d2l

# * NiN块
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )


# * NiN模型
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1), # 标签类别数是10
    nn.AdaptiveAvgPool2d((1, 1)), # 全局平均池化层
    nn.Flatten()) # 将四维的输出转成二维的输出，其形状为(批量大小,10)

# 形状
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# * 训练模型
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

### 含并行连结的网络(GoogLeNet)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171747921.png)
蓝色框用于提取特征信息，白色框用于改变通道数
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171748081.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171809748.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171813095.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171814611.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171815725.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171816075.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171818685.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171818284.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171819166.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410171826042.png)
```python
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

# * Inception块
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    
    def forward(self, x): # 不同路径的前向通路
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1) # 在通道数上这一维度进行连接

# * GoogLeNet模型
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                    Inception(256, 128, (128, 192), (32, 96), 64),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                    Inception(512, 160, (112, 224), (24, 64), 64),
                    Inception(512, 128, (128, 256), (24, 64), 64),
                    Inception(512, 112, (144, 288), (32, 64), 64),
                    Inception(528, 256, (160, 320), (32, 128), 128),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                    Inception(832, 384, (192, 384), (48, 128), 128),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

# 形状
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# * 训练模型
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.waitforbuttonpress()
```

### 批量归一化
若使用Sigmoid激活函数且上一层的输出值较大，会导致这一层梯度较小，梯度在上层(靠近损失)比较大，到下层(靠近数据)比较小(可以看梯度消失的推导)，上层收敛块，下层收敛慢  
下层会抽取、学习一些底层的特征(局部边缘，简单纹理)，上层学习高层语义信息，如果底层信息发生改变会告知上层的权重白学了，我们希望在学习下层时避免变化顶部
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410191514248.png)
假设将分布固定，每一层的输出、梯度都符合某一分布，那么计算梯度会相对稳定，参数更新更加平缓，学习细微的变动也更加容易

**批量归一化**：将不同层的不同位置的小批量(mini-batch)输出的均值和方差固定
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410191516488.png)
参数：
- $B$:小批量 
- $\epsilon$:方差的估计值中有一个小的常量epsilon > 0，为了保证归一化时除数除数不为0(第二行)
- $\gamma$、$\beta$:拉伸参数与偏移参数，可以学习的参数，若当前数值分布不合理，对它们进行限制可以学习得到新的均值和方差，使得数值更加稳定，避免数值变化剧烈

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410191614149.png)
**对全连接层**：输入是二维，每一行是样本，每一列是特征，对每一个特征(每一列)，计算一个标量的均值和方差
**对于卷积层**：每一个像素有多个通道，通道展开可以看做一个向量，这个向量就是这个像素的特征，每一个像素是一个样本，样本数是batch_size\*height\*width，特征是channels

**内部协变量转移**：变量值的分布在训练过程中会发生变化
**随机偏移、随机缩放**：因为每一个小批量都是随机的，所以$\hat{\mu}_B$和$\hat{\sigma}_B$是随机的  
PS：因为加入了噪音，因此可以说它控制了模型的复杂度，由于丢弃法也控制了模型复杂度，因此没有必要跟丢弃法混合使用(类似于数据增强)

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410191640833.png)

**指数加权平均**
- 背景原因：
    - 测试集和训练集的样本分布不一样，使用测试集的均值和方差传入模型，可能得到的结果不太好
    - 测试集可能只有一个样本无法计算均值和标准差
    - 因此需要保存并使用训练过程中的结果来对测试集数据进行归一化

- 解决办法：
这里我们使用***指数加权平均***的方法，先初始化全局均值和方差，再根据每一个batch的均值和方差对全局方差进行更新，更新内容包括*当前batch的均值和方差*并保留*一部分的历史值*，避免出现异常分布，更新方法类似于参数梯度下降算法

- 公式：
    - $moving\_mean = momentum * moving\_mean + (1.0 -momentum) * mean$
    - $moving\_var = momentum * moving\_var + (1.0 -momentum) * var$

```python
import torch
from torch import nn
from d2l import torch as d2l

# * 从零实现批量归一化层
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum): # moving_mean\var 全局均值\方差
    if not torch.is_grad_enabled(): # 当不算梯度的时候(推理)
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps) # 在测试模型的时候，不需要更新参数，直接将输入通过前向传播得到输出
    else:
        assert len(X.shape) in (2, 4) # 确保输入形状为2(全连接层)或者4(卷积层)
        if len(X.shape) == 2:
            mean = X.mean(dim=0) # 全连接层计算每一列的均值
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps) # 每个批量的方差

        '''
        在训练过程中，数据的分布可能会随着模型的学习不断变化（小批量数据的随机性，参数更新，非线性激活函数）。
        通过不断更新移动平均的均值和方差，可以动态地跟踪数据分布的变化趋势。
        在预测阶段，面对新的数据，可以使用相对稳定的均值和方差进行归一化处理，使得模型在不同的数据上表现更加稳定。
        '''
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean # 全局均值的更新
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

# 创建一个正确的BatchNorm图层
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape)) # 有Parameter会自动移动到相应device
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape) # 均值为0 没有Parameter储存在内存CPU
        self.moving_var = torch.ones(shape) # 方差为1

    def forward(self, X):# 当输入X不在内存上将moving_mean/var复制到显存
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y

# * 使用批量归一化层的LeNet
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), 
    BatchNorm(6, num_dims=4), 
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(6, 16, kernel_size=5), 
    BatchNorm(16, num_dims=4), 
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), 

    nn.Flatten(),
    nn.Linear(16*4*4, 120), 
    BatchNorm(120, num_dims=2), 
    nn.Sigmoid(),

    nn.Linear(120, 84), 
    BatchNorm(84, num_dims=2), 
    nn.Sigmoid(),

    nn.Linear(84, 10)
)

lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))

# * 简洁实现
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), 
    nn.BatchNorm2d(6), 
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(6, 16, kernel_size=5), 
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), 
    
    nn.Flatten(),
    nn.Linear(256, 120), 
    nn.BatchNorm1d(120), 
    nn.Sigmoid(),

    nn.Linear(120, 84), 
    nn.BatchNorm1d(84), 
    nn.Sigmoid(),
    nn.Linear(84, 10))

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

### 残差网络(ResNet)
在以前的网络里，更深的网络结构不一定结果会更加好，在ResNet的提出后，残差块的存在可以保证在网络深度增加的同时，精度至少不会低于深度更小的网络，使得很深的网络更加容易训练
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410201543779.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410201550369.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410201551734.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410201552511.png)
ResNet-18结构
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410201633220.png)

**ResNet梯度更新**
如果上层拟合数据较好或者说预测值与真实值差异越小，那么梯度就越小，反向传播乘到最下层时，底下的层的梯度会变得很小(梯度消失)，难以更新。  
ResNet做的事情就是将梯度由单纯的乘法转为加法，在反向传播计算底层的梯度的时候，即使上层的梯度很小(上层参数拟合程度较好)，还可以有底层直接连接过来的梯度的那一部分，从而使得底层的参数更新不会很慢。
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410201732452.png)
```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# * 残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True) # 原地操作节省内存，速度变慢
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X # f(x) = g(x) + x
        return F.relu(Y)

# 输入输出形状一致
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)

# 增加输出通道，减半输出的高和宽
blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)

# * ResNet模型
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block: # 见resnet18结构图
            # 如果不是第一个block，则一个stage由一个1x1conv残差块和一个单独连线的残差块组成
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            # 是第一个block，这个stage由两个单独连线的残差块组成 或者 stage中非第一个残差块的其他残差块
            blk.append(Residual(num_channels, num_channels))
    return blk

# 前两层跟goolenet一样
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 第一个stage通道数输入输出一致
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
# 第二、三、四个stage 每个stage通道数翻倍，高和宽减半
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
# 最后连接全局平均汇聚层与全连接层
net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
)

# 查看形状
X = torch.rand((1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape: ', X.shape)

# * 训练模型
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Chapter 7 : 循环神经网络
### 序列模型
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071653614.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071654744.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071657324.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071709327.png)
f就是机器学习的模型，通过这个模型来预测下一个事件的模型
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071710534.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071713897.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071716523.png)
根据现在观察到的数据和上一个状态的潜变量更新潜变量，减少计算
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071716645.png)
```python
import torch
from torch import nn
from d2l import torch as d2l

# 样本是带噪音的sin函数
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

# * 训练
# 马尔可夫模型
tau = 4 # tao参数
features = torch.zeros((T - tau, tau)) # t-tau是样本数，tau是特征数
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

# 正式训练
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

# * 预测
# 单步预测
# 每次给前四个真实数据，看看后面一个预测的怎么样
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
d2l.plt.show()

# 多步预测
# 只给前四个真实数据，预测第五个，然后用第234个真实数据和第5个预测数据 预测第6个数据
# 这里从600次开始使用多步预测，可以看出错的很离谱(误差的不断累加导致最终结果偏离真实值)
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

# k步预测 给四个点，预测未来的k个点
# e.g. k=16 
# 第一步先给出1234真实点，来预测5，第二步用真实234预测5，来预测6，以此类推16步得到预测20，则预测点第一个点应该t=20
# 第二个点先给出2345真实点，... ，得到t=21的预测值
max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
d2l.plt.show()
```

### 文本预处理
```python
import collections
import re
from d2l import torch as d2l

# * 读取数据集
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 将除了字母以外的所有字符(标点符号，不认识的字母)全部变成空格 简化数据
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

# * 词元化
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word': # 拆分为单词
        return [line.split() for line in lines]
    elif token == 'char': # 拆分为字母
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines, token='char')
for i in range(1):
    print(tokens[i])

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])

# * 词表
# 构建一个字典，将字符串类型的标记映射到从0开始的数字索引中
class Vocab: 
    """文本词表"""
    # min_freq 若一个单词少于这个值则舍弃 reserved_tokens表示句子开始或者句子结束的token
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率从大到小排序token   key:排序的比较  .items()字典转元组 x[1]即词频，元组的第二个元素 
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0 '<unk>'用来表示词汇表中没有的单词
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 字典推导式 将token和id对应起来
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        # 添加高频词
        for token, freq in self._token_freqs:
            if freq < min_freq: # 小于最小出现频率 则舍弃
                break
            if token not in self.token_to_idx: # 检查是否已经存在于词汇表
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1 # 记录该词的索引位置
    
    # 返回token个数
    def __len__(self):
        return len(self.idx_to_token)

    # 给token返回下标
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            # get用于取出字典指定key的value，若没有key默认值是self.unk
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens] # 一直拆分成直至不是list或tuple类型(str)

    # 给下标返回tokens
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        '''
        外层迭代：for line in tokens，表示遍历 tokens 的每一行（即每个子列表）。
                在我们的示例中，这将依次取出 ['I', 'love', 'coding'] 和 ['Python', 'is', 'great']。
        内层迭代：对于每一个 line，再使用 for token in line 遍历每个子列表 line 中的单词（或“token”）。
        生成新列表：对于每一个 token，将其添加到新的列表中。
        '''
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])

# * 整合所有功能
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# corpus是每一个的字符下标 vocab是字表 28是因为实例选择了char参数，返回26个字母+<unk>+空格=28
corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
```

### 语言模型和数据集

### 循环神经网络

### 循环神经网络的从零开始实现

### 循环神经网络的简洁实现

### 通过时间反向传播

## Chapter 8 : 现代循环神经网络
### 门控循环单元(GRU)

### 长短期记忆网络(LSTM)

### 深度循环神经网络

### 双向循环神经网络

### 机器翻译与数据集

### 序列到序列学习(seq2seq)

### 束搜索

## Chapter 9 : 注意力机制
### 注意力提示

### 注意力汇聚：Nadaraya-Watson核回归

### 注意力评分

### Bahdanau 注意力

### 多头注意力

### 自注意力和位置编码

### Transformer

## Chapter 10 : 优化算法
### 优化和深度学习

### 凸性

### 梯度下降

### 随机梯度下降

### 小批量随机梯度下降

### 动量法

### AdaGrad算法

### RMSProp算法

### Adadelta

### Adam算法

### 学习率调度器

## Chapter 11 : 计算性能
### 硬件
**CPU与GPU**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410221614016.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410221615505.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410221616180.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410221617283.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410221618269.png)
**TPU**  
张量处理单元是一种定制化的 ASIC 芯片，它由谷歌从头设计，并专门用于机器学习工作负载。
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410221639761.png)

### 多GPU训练
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410221653630.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410221656742.png)
```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# * 简单网络(LeNet改)
# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# * 数据同步
# 向多个设备分发参数
def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)

# allreduce 将所有向量相加，并将结果广播给所有GPU
def allreduce(data): # 把所有数据放在 GPU:0 上
    for i in range(1, len(data)): # for loop GPU 0 以外的GPU
        data[0][:] += data[i].to(data[0].device) # 把第i个GPU上的data传给第0个tensor的GPU上，再加到 GPU0 对应的数据上
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device) # 将新的结果复制到第i个GPU上

data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i+ 1) for i in range(2)]
print('before allreduce: \n', data[0], '\n', data[1])
allreduce(data)
print('after allresuce: \n', data[0], '\n', data[1])

# * 数据分发
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices) # 根据GPU的个数，将data均匀地切开
print('input', data)
print('load into', devices)
print('output', split)

def split_batch(X, y, devices): # 方便使用，集成起来
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices), nn.parallel.scatter(y, devices))

# * 训练
# 在小批量上实现多GPU训练
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    ls = [loss(lenet(X_shard, device_W), y_shard).sum() for X_shard, y_shard, device_W in zip(X_shards, y_shards, device_params)]
    for l in ls:
        l.backward()
    with torch.no_grad():
        for i in range(len(device_params[0])): # i 层数
            # 对每一个每一个层
            allreduce([device_params[c][i].grad for c in range(len(devices))]) # c GPU个数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 因为每个GPU上参数一致，梯度经过上面的操作也一致，所有在每个GPU上做参数更新

# 定义训练函数
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    device_params = [get_params(params, d) for d in devices] # 复制参数到每一个GPU上

    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 为单个小批量执行多GPU训练
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize() # 同步一次 保证每一个GPU都运算完成
        timer.stop()
        # 在GPU0上评估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
            f'在{str(devices)}')

# 在单GPU上运行
train(num_gpus=1, batch_size=256, lr=0.2)
# 在双GPU上运行
train(num_gpus=2, batch_size=256, lr=0.2)
```

### 多GPU的简洁实现
```python
import torch
from torch import nn
from d2l import torch as d2l

# * 简单网络
def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels, out_channels, num_residuals,
                first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                        nn.Linear(512, num_classes)))
    return net

# * 网络初始化
net = resnet18(10)
devices = d2l.try_all_gpus()

# * 训练
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]

    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    
    # ! nn.DataParallel 在多个GPU上设置模型
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            # 网络被Dataparallel“包装”后，在前向过程会把输入tensor自动分配到每个显卡上。
            # 而Dataparallel使用的是master-slave的数据并行模式，主卡默认为0号GPU，所以在进网络之前，只要移到GPU[0]就可以了
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
            f'在{str(devices)}')

# 单GPU
train(net, num_gpus=1, batch_size=256, lr=0.1)
# 双GPU
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

### 参数服务器
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410241027613.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410241029422.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410241029026.png)

## Chapter 12 : 计算机视觉
### 图像增广
数据增强通过变形数据来获取多样性从而使得模型泛化能力更好
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410241040666.png)
常用操作
- 翻转
  - ![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410241040416.png)
- 切割
  - ![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410241041412.png)
- 颜色
  - ![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410241042990.png)

```python
import torch
import torch.utils
import torch.utils.data
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('./related_data/dog.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5): # num_rows, num_cols 生成图像个数 aug图像增广方法
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()

# * 常用的图像增广方法
# 翻转
# RandomHorizontalFlip() 水平翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())
# RandomVerticalFlip() 垂直翻转
apply(img, torchvision.transforms.RandomVerticalFlip())

# 裁剪
# RandomResizedCrop() 裁剪 size：裁剪后resize的图片大小 scale：指定裁剪区域的面积占原图像的面积的比例范围 ratio：指定裁剪区域的宽高比范围
shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 改变颜色 ColorJitter()
# 改变亮度
apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))
# 改变色调
apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))
# 改变亮度 对比度 饱和度 色调
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# 集合多种图像增广方法 Compose()
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    color_aug, 
    shape_aug
])
apply(img, augs)

# * 使用图像增广进行训练
all_images = torchvision.datasets.CIFAR10(train=True, root='./related_data', download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
d2l.plt.show()

# 只做随机水平翻转
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
test_augs = torchvision.transforms.ToTensor()

# 定义辅助函数，便于读取图像和应用图像增广
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='./related_data', train=is_train, transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            shuffle=is_train, num_wprker=d2l.get_dataloader_workers)
    return dataloader

# * 多GPU训练
def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                                (metric[0] / metric[2], metric[1] / metric[3],
                                None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
            f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
            f'{str(devices)}')

batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) == [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

train_with_data_aug(train_augs, test_augs, net)
```

### 微调(迁移学习)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410250940326.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410250940384.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410250941433.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410250943476.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410250943689.png)
```python
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# ? 热狗识别
# * 获取数据集
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                        'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
d2l.plt.show()

# 因为模型在ImageNet上也做了相同的归一化
normalize = torchvision.transforms.Normalize( 
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
    ])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

# * 定义和初始化模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
print(pretrained_net.fc)
# 修改全连接层
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2) # 输出类别改为2
nn.init.xavier_uniform_(finetune_net.fc.weight)

# * 微调模型
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_lx = [
            param for name, param in net.named_parameters()
            if name not in ['fc.weight', 'fc.bias']]
        
        # FC层的学习率调大，其他层的学习率调小
        trainer = torch.optim.SGD([{
            'params':params_lx},
            {'params':net.fc.parameters(),
            'lr':learning_rate * 10}],
            lr = learning_rate,
            weight_decay=0.001
            )
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 使用小学习率
train_fine_tuning(finetune_net, 5e-5)

# 为了进行比较，所有模型参数初始化成随机值
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

### 目标检测和边界框
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410302053963.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410302056829.png)
```python
import torch
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.plt.imread('related_data/cat_dog.png')
d2l.plt.imshow(img)
d2l.plt.show()

# * 边界框
# 定义两种表示间转换的函数
def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# 猫狗边界框
dog_bbox, cat_bbox = [20.0, 25.0, 180.0, 220.0], [185.0, 55.0, 290.0, 210.0]

boxes = torch.tensor((dog_bbox, cat_bbox))
print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

# 将边界框画出
def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]),
                            width=bbox[2] - bbox[0],
                            height=bbox[3] - bbox[1],
                            fill=False,
                            edgecolor=color,
                            linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show()
```

### 锚框
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410311307231.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410311308790.png)
将框住的范围与真实物体框求雅可比值，再设置一个雅可比值，小于则为负类(背景)，大于则为正类(关联)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410311320571.png)
训练阶段：筛选出一部分锚框，每个真实框对应不止一个锚框，每个锚框对应了一个样本(这里生成9个训练样本，四个真实框)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410311326030.png)
预测阶段：筛选最符合的锚框
NMS：把同类别取概率最大值，去掉重复面积大于某个值的所有框
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410311341637.png)

**锚框的流程**：
1. 生成许多锚框
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410311427291.jpg)
2. 用真实框(ground truth)框去标记，计算`步骤1`中所有的锚框。
   - 标记方法：计算所有生成锚框与真实框的IoU值，形成一个表格，先进行如上图的筛选操作，给每一个真实框都分配给一个锚框后，遍历剩下锚框的IoU的最大值，小于设定阈值为背景，大于的选一个最接近真实框的
   - 计算两框交集左上角的广播方法
    ![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410311815739.jpg)
3. 对标记好的所有锚框预测偏移量，背景类偏移为0
    - 偏移量设置： ![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202410312050593.png)
4. 用NMS合并属于同一类别的类似的预测边界框，选出预测最好的，把没我预测的好的框删除，简化输出
5. 最终得到每一个对象或物体的一个最终预测边框


```python
import torch
from d2l import torch as d2l

# * 生成多个锚框
'''
对于每一个像素，生成高宽都不同的锚框，锚框宽度w*s*sqrt(r) 这里w，h是图像宽和高，s是缩放比，r是宽高比
我们设置许多s取值 s1,s2...sm 和许多r取值 r1,r2...rm，为了简化样本个数，我们只考虑包含r1或s1的组合
即(s1, r1),(s1, r2),...(s1, rm),(s2, r1),(s3, r1)...(sm, r1)
'''
def multibox_prior(data, sizes, ratios): # data就是input
    in_height, in_width = data.shape[-2:] # 图像高和宽，像素点个数
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # 锚框个数 如上所述的组合个数
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素中心，设置偏移量，因为一个像素得分高宽均为1，因此偏移至中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height # 在y轴上缩放步长
    steps_w = 1.0 / in_width # 在x轴上缩放步长

    # 生成所有锚框的中心点
    '''
    我们希望锚框的中心位置对准像素的中心而非像素的左上角
    torch.arange(in_height, device=device) + offset_h 找到每一个像素点的中心坐标

    通过归一化操作，可以避免因输入图像尺寸变换而对结果的影响，也方便torch框架进行运算
    (torch.arange(in_height, device=device) + offset_h) * steps_h
    '''
    ceter_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    ceter_w = (torch.arange(in_width, device=device) + offset_w) * steps_w

    '''
    torch.meshgrid() 接受多个一维张量作为输入，并根据指定的索引模式 生成相应的多维网格张量。
    当 indexing='ij' （默认）时，第一个输入张量沿着行方向扩展，第二个输入张量沿着列方向扩展，行数是第一个输入张量的元素个数，列数是第二个输入张量的元素个数
    当 indexing='xy' 时，第一个输入张量沿着列方向扩展，第二个输入张量沿着行方向扩展，行数是第二个输入张量的元素个数，列数是第一个输入张量的元素个数
    
    通过meshgrid操作可以得到每一个锚框的中心点 应该在y轴和x轴上的偏移，拓展成了一个矩阵，shape(in_height, in_width)
    
    test.py
    import torch
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5])
    # 默认 indexing='ij'
    xx, yy = torch.meshgrid(x, y)
    print(xx)
    print(yy)
    # indexing='xy'
    xx_xy, yy_xy = torch.meshgrid(x, y, indexing='xy')
    print(xx_xy)
    print(yy_xy)
    '''
    shift_y, shift_x = torch.meshgrid(ceter_h, ceter_w, indexing='ij')
    # 拉成一维
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成每个像素点上 `boxes_per_pixel` 个锚框的高和宽
    # 将s和r组合 取出r1和s1,s2...sm s1和r2,r3...rm做组合操作
    # 为了适应不同图像的宽高比，消除原图像wh的影响，因此进行了归一化操作，
    '''
    先不看 in_height / in_width。上面解得的归一化后的公式与代码所写的一模一样。
    代码中的和就是锚框归一化后的宽高（此时消除了原图像 w 和 h 的影响，
    可以认为，r 所代表的宽高比就是此时锚框的宽高比，r=1 时，是一个正方形锚框，也即此时和的值是一样的）。
    但是呢，由于我们显示的时候需要乘以图像的实际宽高,所以，乘后的锚框实际宽高比就不是 1 了，所以才要乘以 in_height / in_width，
    作用就是抵消乘以实际图像长宽后 r 会改变的问题，当然这样做存粹是为了显示方便（也让你误以为 r 是指锚框的宽高比），
    带来的副作用就是，锚框的实际面积就不再是原始图像的。

    由于实际在我们进行目标检测时，特征图长和宽都是相同的，比如 (19, 19)、(7, 7)，
    所以 in_height / in_width 恒等于 1，因此对于实际的使用并不会带来副作用。
    但此时，如果要将锚框显示出来，归一化后的锚框再乘以图像实际长宽后，所显示的锚框的长宽比会改变。
    in_height / in_width 这部分失效了。好消息是，面积是原图的，又符合定义了。
    '''

    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # 获得半高和半宽
    '''
    torch.stack((-w, -h, w, h))  计算了相较锚点中心的偏移量，方便计算(x1, y1, x2, y2)

    .T 是将偏移量的形状进行转置，使得每行对应一个锚框的四个边界偏移量 (xmin, ymin, xmax, ymax)

    torch.repeat:输入一维张量，参数为两个(m,n)，即表示先在列上面进行重复n次，再在行上面重复m次，输出张量为二维
    repeat(in_height * in_width, 1)  因为每一个像素点都需要计算偏移，所以进行复制

    最后除以2 算得距离每一个锚框中心偏移量
    '''
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 
    '''
    torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1) 
    得到形状为 [in_height * in_width, 4] 的矩阵 每个像素点中心点的偏移
    .repeat_interleave(boxes_per_pixel, dim=0)：将每个像素点的中心坐标重复 boxes_per_pixel 次
    在dim=0上进行复制，这个一个像素点后紧跟着着这个像素点 boxes_per_pixel 个的锚框的中心偏移

    test.py
    a = torch.randn(2,2)
    print(a,a.repeat_interleave(3,dim=0)) # 重复三次
    output:
    tensor(
        [[-0.4402,  0.8147],
        [ 0.0875, -1.5945]])
    tensor(
        [[-0.4402,  0.8147],
        [-0.4402,  0.8147],
        [-0.4402,  0.8147],
        [ 0.0875, -1.5945],
        [ 0.0875, -1.5945],
        [ 0.0875, -1.5945]])
    '''
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    # 结合像素中心坐标和锚框偏移 每个锚框的四角坐标 (xmin, ymin, xmax, ymax)，也就是锚框的实际坐标
    output = out_grid + anchor_manipulations
    # 添加了一个维度 batch_size , shape [1, in_height * in_width * boxes_per_pixel, 4]
    return output.unsqueeze(0)

# 查看返回形状
img = d2l.plt.imread('related_data/cat_dog.png')
h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)

# 访问(200, 200)中心的第一个锚框
boxes = Y.reshape(h, w, 5, 4)
print(boxes[200, 200, 0, :])

# 显示以图像中某个像素为中心的所有锚框
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

# 以100，100为中心的所有锚框
d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[100, 100, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
d2l.plt.show()

# * 交并比IoU
#@save
def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    # None的作用增加了一个维度 便于广播
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # 使用 .clamp(min=0) 是为了处理可能的负值情况（即两个框没有重叠时），确保交集的宽和高至少为 0。
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    # torch.nonzero() 返回一个包含输入 input 中非零元素索引的张量.输出张量中的每行包含 input 中非零元素的索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    # 若为True，则保留当前位置的值，若为False，则舍弃
    box_j = indices[max_ious >= iou_threshold]
    # ! 这一步就是保底操作，能用的锚框都可以有一个最接近的真实框对应，小于阈值的直接舍弃，不给这些锚框对应标号
    anchors_bbox_map[anc_i] = box_j

    # 用于标记需要忽略的列（真实框）置-1
    col_discard = torch.full((num_anchors,), -1)
    # 用于标记需要忽略的行（锚框）置-1
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        # 全局最大值索引，是标量
        max_idx = torch.argmax(jaccard)
        # 确定真实框索引和锚框索引
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        # 更新分配映射
        # ! 这一步是图中的表格算法操作 也是确保每一个真实框都至少有一个对应锚框
        anchors_bbox_map[anc_idx] = box_idx
        # 将本行本列置0
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        # 调用 offset_boxes 计算锚框到真实边界框的偏移量，并乘以 bbox_mask 以掩盖无效锚框。
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)

    # 返回锚框与真实框的偏移，锚框是否为背景，对应的类别标号
    return (bbox_offset, bbox_mask, class_labels)

ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
d2l.plt.show()

labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))

# * 使用非极大值抑制预测边界框
def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

# NMS按降序对置信度排序返回索引
def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
d2l.plt.show()

output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(output)

fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
d2l.plt.show()
```

### 多尺度目标检测
```python
import torch
from d2l import torch as d2l

# * 多尺度锚框
# 为了减少锚框数量，避免一些重复锚框，用较小的锚框可以检测小的物体，采样更多区域，用较大的锚框可以检测较大的物体，采样更少区域
img = d2l.plt.imread('related_data/cat_dog.png')
h, w = img.shape[:2]
print(h, w)

def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 生成特征图 前两个维度上的值不影响输出 
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    # 在特征图上每个像素点中心生成锚框
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    # anchors[0] 就是batchsize 
    # 乘以bbox_scale 就是在还原锚框在原图像的比例，以原图像的尺寸去还原特征图上的锚框，从而完成采样操作
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)
    d2l.plt.show()

# 探测小目标 假设特征图长宽是4
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
# 探测中目标 假设特征图长宽是2
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
# 探测大目标 假设特征图长宽是1
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

### 目标检测数据集
```python
import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

# * 下载香蕉检测数据集
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

# * 读取数据集
#@save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    # iterrows() 逐行处理数据
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))

    '''
    torch.unsqueeze(dim) 在第dim个维度增加一个维度
    维度组成：batch_size, num_object, feature(类别和四个框坐标) e.g.[32, 1, 5]
    这里的targets是边缘框的四个坐标，除以256归一化，方便框架输入，类似ToTensor()
    增加一个维度(位置在第1维度即目标个数)表示一个图片中最多出现的物体数，这里只有香蕉所以是1
    '''
    return images, torch.tensor(targets).unsqueeze(1) / 256

# 创建自定义Dataset实例
class BananaDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read' + str(len(self.features)) +
                            (f' training examples' if is_train else f'validation examples'))
    
    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])
    
    def __len__(self):
        return len(self.features)

# 为训练集和测试集返回两个数据加载器实例
def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananaDataset(is_train=True), batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananaDataset(is_train=False), batch_size)
    return train_iter, val_iter

# 查看shape
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)

# * 演示
# permute 维度转置 通道数放在最后
# 除以255 归一化 用于显示 库的要求
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)

for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
d2l.plt.show()
```

### 单发多框检测(SSD)
***SSD***
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411011630820.png)
- 输入图像之后，首先进入一个基础网络来抽取特征，抽取完特征之后对每个像素生成大量的锚框（每个锚框就是一个样本，然后预测锚框的类别以及到真实边界框的偏移）
- SSD 在给定锚框之后直接对锚框进行预测，而不需要做两阶段（为什么 Faster RCNN 需要做两次，而 SSD 只需要做一次？SSD 通过做不同分辨率下的预测来提升最终的效果，越到底层的 feature map，就越大，越往上，feature map 越少，因此底层更加有利于小物体的检测，而上层更有利于大物体的检测）
- SSD 不再使用 RPN 网络，而是直接在生成的大量样本（锚框）上做预测，看是否包含目标物体；如果包含目标物体，再预测该样本到真实边缘框的偏移

***YOLO***
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411011637273.png)
- yolo 也是一个 single-stage 的算法，只有一个单神经网络来做预测
- yolo 也需要锚框，这点和 SSD 相同，但是 SSD 是对每个像素点生成多个锚框，所以在绝大部分情况下两个相邻像素的所生成的锚框的重叠率是相当高的，这样就会导致很大的重复计算量。
- yolo 的想法是尽量让锚框不重叠：首先将图片均匀地分成 S * S 块，每一块就是一个锚框，每一个锚框预测 B 个边缘框（考虑到一个锚框中可能包含多个物体），所以最终就会产生 S ^ 2 * B 个样本，因此速度会远远快于 SSD
- yolo 在后续的版本（V2,V3,V4...）中有持续的改进，但是核心思想没有变，真实的边缘框不会随机的出现，真实的边缘框的比例、大小在每个数据集上的出现是有一定的规律的，在知道有一定的规律的时候就可以使用聚类算法将这个规律找出来（给定一个数据集，先分析数据集中的统计信息，然后找出边缘框出现的规律，这样之后在生成锚框的时候就会有先验知识，从而进一步做出优化）

***center net***
- 基于非锚框的目标检测
- center net 的优点在于简单
- center net 会对每个像素做预测，用 FCN 对每个像素做预测（类似于图像分割中用 FCN 对每个像素标号），预测该像素点是不是真实边缘框的中心点（将目标检测的边缘框换算成基于每个像素的标号，然后对每个像素做预测，就免去了一些锚框相关的操作）

SSD的实现思路：
1. 生成一堆锚框
2. 根据真实标签给每个锚框打标签(类别、偏移、mask)
3. 模型为每个锚框做一个预测(类别、偏移)
4. 计算上述二者的差异损失，更新权重参数

```python
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# * 模型
# 类别预测层 预测锚框类别
# 目标是n类，锚框总共有n+1个类别，其中0类是背景
def cls_predictor(num_inputs, num_anchors, num_classes): # 输入图像输入尺寸，锚框个数，目标类别个数
    # 返回num_anchors * (num_classes + 1) 对应每一个锚框对每一种类别的预测值
    # 此卷积层的输入和输出的宽度和高度保持不变 类似于卷积层只是参数更加少
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

# 边界框预测层 预测锚框与真实框的偏移
def bbox_predictor(num_inputs, num_anchors):
    # 返回num_anchors * 4 给每个锚框预测4个偏移量
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# 连接多尺度的预测
def forward(x, block):
    return block(x)

# 生成2个特征图 高宽是20 每个中心点生成5个锚框 类别是10
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape, Y2.shape)
# (torch.Size([2, 55, 20, 20]) 第一个尺度每一个像素点生成55个预测值 11*5 总共20*20个像素点
# torch.Size([2, 33, 10, 10])) 第二个尺度每一个像素点生成335个预测值 11*3 总共10*10个像素点

# 连接除了批量大小以外的所有参数
def flatten_pred(pred):
    # 先把通道数放在最后，把每个像素的每个锚框预测的类别放在一起
    # 从第一个维度开始执行展平操作，shape变为（批量大小，高*宽*通道数）
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

# 连接不同尺度下的框
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# 尽管Y1和Y2在通道数、高度和宽度方面具有不同的大小，我们仍然可以在同一个小批量的两个不同尺度上连接这两个预测输出
print(concat_preds([Y1, Y2]).shape)

# 高和宽减半块 下采样块 变换通道数
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)

# 基本网络块
def base_net():
    blk = []
    # 通道数从3开始，再到16，再到32，再到64 同时高宽也减半三次
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)
print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)

# 完整的模型 五个模块构成
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        # 将高度和宽度压到1
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

# 给每一个块定义前向计算
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    # 生成当前fmap的卷积层输出
    Y = blk(X)
    # 生成当前fmap的锚框
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    # 生成类别和偏移预测
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

# 超参数
# 最下层就是fmap尺寸较大的用小的锚框，后面锚框尺寸越来越大
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        # 每个Stage输出通道数
        idx_to_in_channels = [64, 128, 128, 128, 128]
        # 做五次预测
        for i in range(5):
            # setattr 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        # 把类别作为最后一维拿出来 方便预测
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape) # 所有stage每个像素每个anchor之和
print('output class preds:', cls_preds.shape) # 32 batchsize 2 类别加1 
print('output bbox preds:', bbox_preds.shape) # anchor*4 每个锚框与真实框的偏移

# * 训练模型
# 读取数据集
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

# 初始化参数
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 定义损失函数和评价函数
cls_loss = nn.CrossEntropyLoss(reduction='none') # 类别损失
bbox_loss = nn.L1Loss(reduction='none') # 偏移损失 选择L1的原因：如果预测与真实差距特别大，不会返回一个特别大的损失

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1) # mask标注了背景框和非背景框
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# 正式训练
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

# 保存参数
torch.save(net.state_dict(), 'related_data/SSD.pth')
print("Model parameters saved to SSD.pth")

# * 预测目标
X = torchvision.io.read_image('related_data/banana.jpeg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

# Load model parameters before testing
net_load = TinySSD(num_classes=1).to(device)  # Make sure to initialize the model first
net_load.load_state_dict(torch.load('related_data/SSD.pth'))
print("Model parameters loaded from SSD.pth")

def predict(X):
    net_load.eval()
    anchors, cls_preds, bbox_preds = net_load(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)

def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    d2l.plt.show()

display(img, output.cpu(), threshold=0.9)
```

### 区域卷积神经网络(R-CNN)系列
***R-CNN***
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411011518647.png)
**R-CNN 模型的四个步骤**：
1. 对输入图像使用选择性搜索来选取多个高质量的提议区域。这些提议区域通常是在多个尺度下选取的，并具有不同的形状和大小；每个提议区域都将被标注类别和真实边框
2. 选择一个预训练的卷积神经网络，并将其在输出层之前截断。将每个提议区域变形为网络需要的输入尺寸(ROI Pooling)，并通过前向传播输出抽取的提议区域特征
3. 将每个提议区域的特征连同其标注的类别作为一个样本。训练多个支持向量机对目标分类，其中每个支持向量机用来判断样本是否属于某一个类别
4. 将每个提议区域的特征连同其标注的边界框作为一个样本，训练线性回归模型来预测真实边界框

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411011520862.png)
**RoI pooling（兴趣区域池化层）**
- R-CNN 中比较关键的层，作用是将大小不一的锚框变成统一的形状
- 给定一个锚框，先将其均匀地分割成 n * m 块，然后输出每块里的最大值，这样的话，不管锚框有多大，只要给定了 n 和 m 的值，总是输出 nm 个值，这样的话，不同大小的锚框就都可以变成同样的大小，然后作为一个小批量，之后的处理就比较方便了
- 上图中对 3 * 3 的黑色方框中的区域进行 2 * 2 的兴趣区域池化，由于 3 * 3 的区域不能均匀地进行切割成 4 块，所以会进行取整（最终将其分割成为 2 * 2、1 * 2、2 * 1、1 * 1 四块），在做池化操作的时候分别对四块中每一块取最大值，然后分别填入 2 * 2 的矩阵中相应的位置

总结：尽管 R-CNN 模型通过预训练的卷积神经网络有效地抽取了图像特征，但是速度非常慢（如果从一张图片中选取了上千个提议区域，就需要上千次的卷积神经网络的前向传播来执行目标检测，计算量非常大）  
R-CNN 每次拿到一张图片都需要抽取特征，如果说一张图片中生成的锚框数量较大，抽取特征的次数也会相应的增加，大大增加了计算量因此，R-CNN 的主要性能瓶颈在于，对于每个提议区域，卷积神经网络的前向传播是独立的，没有共享计算

***Fast R-CNN***
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411011540172.png)
- Fast R-CNN 的改进是：在拿到一张图片之后，首先使用 CNN 对图片进行特征提取（不是对图片中的锚框进行特征提取，而是对整张图片进行特征提取，仅在整张图像上执行卷积神经网络的前向传播），最终会得到一个 7 * 7 或者 14 * 14 的 feature map
- 抽取完特征之后，再对图片进行锚框的选择（selective search），搜索到原始图片上的锚框之后将其（按照一定的比例）映射到 CNN 的输出上(图中 CNN输出即蓝色区域 中的 红色框即比例映射的锚框)
- 映射完锚框之后，再使用 RoI pooling 对 CNN 输出的 feature map 上的锚框进行特征抽取，生成固定长度的特征，即绿色长条向量（将 n * m 的矩阵拉伸成为 nm 维的向量），之后再通过一个全连接层（这样就不需要使用SVM一个一个的操作，而是一次性操作了）对每个锚框进行预测：物体的类别和真实的边缘框的偏移
- 上图中黄色方框的作用就是将原图中生成的锚框变成对应的向量
- Fast R-CNN 相对于 R-CNN 更快的原因是：Fast R-CNN 中的 CNN 不再对每个锚框抽取特征，而是对整个图片进行特征的提取（这样做的好处是：不同的锚框之间可能会有重叠的部分，如果对每个锚框都进行特征提取的话，可能会对重叠的区域进行多次重复的特征提取操作），然后再在整张图片的feature中找出原图中锚框对应的特征，最后一起做预测

***Faster R-CNN***
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411011555113.png)
- Faster R-CNN 提出将选择性搜索替换为区域提议网络（region proposal network，RPN），模型的其余部分保持不变，从而减少区域的生成数量，并保证目标检测的精度
- Faster R-CNN 的改进：使用 RPN 神经网络来替代 selective search 
- RoI 的输入是CNN 输出的 feature map 和生成的锚框
- RPN 的输入是 CNN 输出的 feature map，输出是一些比较高质量的锚框（可以理解为一个比较小而且比较粗糙的目标检测算法： CNN 的输出进入到 RPN 之后再做一次卷积，然后生成一些锚框（可以是 selective search 或者其他方法来生成初始的锚框），再训练一个二分类问题：预测锚框是否框住了真实的物体以及锚框到真实的边缘框的偏移，最后使用 NMS 进行去重，使得锚框的数量变少）
- RPN 的作用是生成大量结果很差的锚框，然后进行预测，最终输出比较好的锚框供后面的网络使用（预测出来的比较好的锚框会进入 RoI pooling，后面的操作与 Fast R-CNN 类似）
- 通常被称为两阶段的目标检测算法：RPN 做小的目标检测（粗糙），整个网络再做一次大的目标检测（精准）
- Faster R-CNN 目前来说是用的比较多的算法，准确率比较高，但是速度比较慢

***Mask R-CNN***
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411011601851.png)
- 如果在训练集中还标注了每个目标在图像上的像素级位置，Mask R-CNN 能够有效地利用这些相近地标注信息进一步提升目标检测地精度
- 假设有每个像素的标号的话，就可以对每个像素做预测（FCN）
- 将兴趣区域汇聚层替换成了兴趣区域对齐层（RoI pooling -> RoI align），使用双线性插值（bilinear interpolation）保留特征图上的空间信息，进而更适于像素级预测：对于pooling来说，假如有一个3 * 3的区域，需要对它进行2 * 2的RoI pooling操作，那么会进行取整从而切割成为不均匀的四个部分，然后进行 pooling 操作，这样切割成为不均匀的四部分的做法对于目标检测来说没有太大的问题，因为目标检测不是像素级别的，偏移几个像素对结果没有太大的影响。但是对于像素级别的标号来说，会产生极大的误差；RoI align 不管能不能整除，如果不能整除的话，会直接将像素切开，切开后的每一部分是原像素的加权（它的值是原像素的一部分）
- 兴趣区域对齐层的输出包含了所有与兴趣区域的形状相同的特征图，它们不仅被用于预测每个兴趣区域的类别和边界框，还通过额外的全卷积网络预测目标的像素级位置

**总结**
- R-CNN 是最早、也是最有名的一类基于锚框和 CNN 的目标检测算法（R-CNN 可以认为是使用神经网络来做目标检测工作的奠基工作之一），它对图像选取若干提议区域，使用卷积神经网络对每个提议区域执行前向传播以抽取其特征，然后再用这些特征来预测提议区域的类别和边框
- Fast/Faster R-CNN持续提升性能：Fast R-CNN 只对整个图像做卷积神经网络的前向传播，还引入了兴趣区域汇聚层（RoI pooling），从而为具有不同形状的兴趣区域抽取相同形状的特征；Faster R-CNN 将 Fast R-CNN 中使用的选择性搜索替换为参与训练的区域提议网络，这样可以在减少提议区域数量的情况下仍然保持目标检测的精度；Mask R-CNN 在 Faster R-CNN 的基础上引入了一个全卷积网络，从而借助目标的像素级位置进一步提升目标检测的精度
- Faster R-CNN 和 Mask R-CNN 是在追求高精度场景下的常用算法（Mask R-CNN 需要有像素级别的标号，所以相对来讲局限性会大一点，在无人车领域使用的比较多）

### 语义分割和数据集
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411061555132.png)
语义分割的应用场景：  背景虚化
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411061556631.png)
```python
import os
import torch
import torchvision
from d2l import torch as d2l

# * Pascal VOC2012 语义分割数据集
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)

# 绘制前五个输入图像和标签
# 在标签图像中，白色和黑色分别表示边框和背景，而其他颜色则对应不同的类别
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n)
d2l.plt.show()

# 列举RGB颜色值和类名
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 每个像素的索引
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

y = voc_label_indices(train_labels[0], voc_colormap2label())
print(y[105:115, 130:140], VOC_CLASSES[1])

# 预处理
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像 保证随机裁剪后图像和标签一一对应"""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width)) # 返回裁剪的框
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
d2l.plt.show()

# 自定义语义分割数据集类
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)
    
    # 假设图片比crop_size还要小，直接舍弃 
    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    # 使用索引方法 返回经过randomcrop的图像
    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)

batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape) # 
    print(Y.shape) # 每个像素的类别序号
    break

# 整合
def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

### 转置卷积
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411061819463.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411061823426.png)
**转置称谓的来源**
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411062012804.jpg)
1. 对于卷积 Y = X * W
    - " * "代表卷积
    - 可以对 W 构造一个 V （V 是一个比较大的向量），使得卷积等价于矩阵乘法 Y‘ = VX’
    - 这里的 Y‘， X’ 是 Y， X 对应的向量版本（将 Y， X 通过逐行连结拉成向量）
    - 如果 X’ 是一个长为 m 的向量，Y‘ 是一个长为 n 的向量，则 V 就是一个 n×m 的矩阵
2. 转置卷积同样可以对 W 构造一个 V ，则等价于 Y‘ = VTX'
   - 按照上面的假设 VT 就是一个  m×n ，则 X’ 就是一个长为 n 的向量，Y‘ 就是一个长为 m 的向量，X 和 Y 的向量发生了交换
   - 从 V 变成了 VT 所以叫做转置卷积

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411062205126.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411062209663.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411062211993.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411062217366.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411062215405.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411062216306.png)
```python
import torch
from torch import nn
from d2l import torch as d2l

# * 基本操作
#  以步幅为1滑动卷积核窗口，结果是一个(nh+kh-1,nw +hw-1)的张量
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1), X.shape[1] + w - 1) # output
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i + h, j:j + w] += X[i, j] * K
    return Y

# 验证
X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K) )

# 使用高级API获得相同的结果
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))

# * 填充、步幅、多通道
# 这里的填充实际上是裁剪，padding等于1，对结果上下左右裁剪一圈
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X)) # (2xh + 2kh - 2s) * 2s

# 多通道
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape == X.shape)

# * 与矩阵变换的联系
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
print(Y)

def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
print(W)

print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))
Z = trans_conv(Y, K)
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))
```

### 全连接卷积神经网络
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071336150.png)
K个通道对应不同的类别，对(224，224)中每个像素点做预测  
初始化卷积核参数的方法：双线性插值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071421938.png)
```python
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# * 构造模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
# 查看后三层的网络结构
print(list(pretrained_net.children())[-3:])

# 创建一个全卷积网络实例net
net = nn.Sequential(*list(pretrained_net.children())[:-2]) # 去掉全局池化层和全连接层
X = torch.rand(size=(1, 3, 320, 480))
print(net(X).shape)

num_classes = 21 # pascal voc2012数据集类别数
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                    kernel_size=64, padding=16, stride=32))

# * 初始化转置卷积层
# 使用经过双线性插值初始化的核，再使用转置卷积层，可以实现双线性插值的上采样
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

# 实验
# 输出图像的高宽翻倍
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
img = torchvision.transforms.ToTensor()(d2l.Image.open('related_data/cat_dog.png').convert('RGB'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0))
d2l.plt.show()
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img)
d2l.plt.show()

# 用双线性插值的上采样初始化转置卷积层，对于卷积层使用Xavier初始化参数
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# * 读取数据集
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

# * 训练
def loss(inputs, targets):
    # 计算损失的结果是三维矩阵(样本，高，宽) 计算高和宽的矩阵就可以得到一个一维的值(每个样本的损失)
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# * 预测
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    # argmax(dim=1) 在每一个像素有21个通道(类别) 找个通道上最大的值即预测类别
    pred = net(X.to(devices[0])).argmax(dim=1)
    # reshape成跟输入图像尺寸一样
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
d2l.plt.show()
```

### 风格迁移
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071529306.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071531462.png)
`gram矩阵`的计算过程与意义：可以获取不同通道间风格的相似性
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202411071611836.png)
```python
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# * 阅读内容和风格图像
d2l.set_figsize()
content_img = d2l.Image.open('related_data/content_img.jpeg')
d2l.plt.imshow(content_img)
d2l.plt.show()

style_img = d2l.Image.open('related_data/style_img.jpeg')
d2l.plt.imshow(style_img)
d2l.plt.show()

# * 预处理和后处理
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

# 预处理函数preprocess对输入图像在RGB三个通道分别做标准化，并将结果变换成卷积神经网络接受的输入格式
def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

# 后处理函数postprocess则将输出图像中的像素值还原回标准化之前的值
def postprocess(img):
    img = img[0].to(rgb_std.device)
    # 由于图像打印函数要求每个像素的浮点数值在0～1之间，我们对小于0和大于1的值分别取0和
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# * 抽取图像特征
pretrained_net = torchvision.models.vgg19(pretrained=True)

# 哪些层的输出用来匹配样式，哪些层用来匹配内容
# 网络越靠下，越匹配局部信息，越往上，越匹配全局信息，样式二者均想要保留，而内容允许局部信息更新(变形)，仅保留全局的大致的样子
style_layers, content_layers = [0, 5, 10, 19, 28], [25]

# 网络仅保留28层结构
net = nn.Sequential(*[pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)])

# 保留内容层和风格层的输出
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# get_contents函数对内容图像抽取内容特征
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

# get_styles函数对风格图像抽取风格特征
def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

# * 定义损失函数
# 内容损失
def content_loss(Y_hat, Y):
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。
    return torch.square(Y_hat - Y.detach()).mean()

# 风格损失 
# 风格一致的定义不是说输出图像要和样式图像的像素值一致，而是每个通道内的统计分布与通道间的统计分布和样式图像一致
def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# 全变分损失
# 去除合成图像里的高频噪点(特别亮或者特别暗的像素点) 使每个像素能与临近的像素值相似
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

# 损失函数
# 风格转移的损失函数是内容损失、风格损失和总变化损失的加权和。 
# 通过调节这些权重超参数，我们可以权衡合成图像在保留内容、迁移风格以及去噪三方面的相对重要性。
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

# * 初始化合成图像
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        # 权重初始化为图像形状的随机值
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

# 训练模型
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X

device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
```

### 实战Kaggle比赛：图像分类(CIFAR-10)
```python
import collections
import math
import os
import shutil
import pandas as pd
import torch
import torch.utils
import torch.utils.data
import torchvision
from torch import nn
from d2l import torch as d2l

# * 获取并组织数据集
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# 如果使用完整的Kaggle竞赛的数据集，设置demo为False
demo = True

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar-10/'

#@save
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))

# 将验证集从原始的训练集中拆分出来
def copyfile(filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                    'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                        'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                        'train', label))
    return n_valid_per_label

# 在预测期间整理测试集
def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                                'unknown'))

# 调用前面定义的函数
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)

# * 图像增广
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010])
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010])
])

# * 读取数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),
                                                            transform=transform_train) for folder in ['train', 'train_valid']]
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

# 指定定义的所有图像增广
train_iter, train_valid_iter = [
    torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True
    )
    for dataset in (train_ds, train_valid_ds)
]

valid_iter = torch.utils.data.DataLoader(
    valid_ds, batch_size, shuffle=False, drop_last=True
)
test_iter = torch.utils.data.DataLoader(
    test_ds, batch_size, shuffle=False, drop_last=True
)

# * 模型
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction='none')

# * 训练函数
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')

# * 训练和验证模型
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# * 用测试集进行分类并提交结果
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
    lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('related_data/submission.csv', index=False)
```

### 实战Kaggle比赛：狗的品种识别(ImageNet Dogs)
```python
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# * 获取和整理数据集
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# 如果使用Kaggle比赛的完整数据集，请将下面的变量更改为False
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')

# 整理数据集(同上一章)
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)

# * 图像增广
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(
        224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
    ),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4
    ),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

# * 读取数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)

# * 微调预训练模型
# 使用预训练的模型，固定特征提取层的参数
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    # 冻结参数
    for param in finetune_net.features.parameters(): # 特征提取层，不包括全连接层，有卷积层
        param.requires_grad = False
    return finetune_net

# 计算损失
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')

# * 定义训练函数
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # 只训练小型自定义输出网络
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')

# * 训练和验证模型
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# * 对测试集分类并提交结果
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
        lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

## Chapter 13 : 自然语言处理: 预训练
### 词嵌入(word2vec)

### 近似训练

### 用于与训练词嵌入的数据集

### 预训练word2vec

### 全局向量的词嵌入(GloVe)

### 子词嵌入

### 词的相似性和类比任务

### 来自Transformers的双向编码器表示(BERT)

### 用于预训练BERT的数据集

### 预训练BERT

## Chapter 14 : 自然语言处理: 应用
### 情感分析及数据集

### 情感分析：使用循环神经网络

### 情感分析：使用卷积神经网络

### 自然语言推断与数据集

### 自然语言推断：使用注意力

### 针对序列级和词元级应用微调BERT

### 自然语言推断：微调BERT

