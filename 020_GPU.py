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