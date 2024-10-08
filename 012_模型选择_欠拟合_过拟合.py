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