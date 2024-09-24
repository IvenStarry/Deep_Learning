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