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