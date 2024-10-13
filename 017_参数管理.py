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