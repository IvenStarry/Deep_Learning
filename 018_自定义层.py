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