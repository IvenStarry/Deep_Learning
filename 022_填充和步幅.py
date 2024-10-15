import torch
from torch import nn

# * 填充
def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape) # 元组的连接
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:]) # 去掉前两维 只看宽高

# 填充固定高度宽度
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1) # 上下左右各填充1行或1列
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape) # 8-3+1*2+1

# 填充不同的高度和宽度
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape) # (8-5+2*2+1, 8-3+1*2+1)

# * 步幅
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape) # (8-3+1*2+2)/2 向下取整

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape) # ((8-3+0*2+3)/3, (8-5+1*2+4)/4) 向下取整