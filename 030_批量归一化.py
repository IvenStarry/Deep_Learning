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