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