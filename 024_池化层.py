import torch
from torch import nn
from d2l import torch as d2l

# * 最大池化层和平均池化层
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] =  X[i:i + p_h, j:j +p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

# 验证
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'avg'))

# * 填充和步幅
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
pool2d = nn.MaxPool2d(3) # torch框架中步幅与池化窗口大小一致 (4-3+0+3)/3

# 手动设定填充和步幅
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X)) # (4-3+1*2+2)/2

# 设定任意大小池化窗口
pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
print(pool2d(X)) # ((4-2+1*2+2)/2, (4-3+1*2+3)/3)

# * 多个通道
X = torch.cat((X, X + 1), 1) # 在channel即第一维度上进行叠加
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))