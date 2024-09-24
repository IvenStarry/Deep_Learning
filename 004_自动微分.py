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