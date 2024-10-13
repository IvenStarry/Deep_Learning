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