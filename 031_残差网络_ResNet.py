import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# * 残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True) # 原地操作节省内存，速度变慢
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X # f(x) = g(x) + x
        return F.relu(Y)

# 输入输出形状一致
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)

# 增加输出通道，减半输出的高和宽
blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)

# * ResNet模型
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block: # 见resnet18结构图
            # 如果不是第一个block，则一个stage由一个1x1conv残差块和一个单独连线的残差块组成
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            # 是第一个block，这个stage由两个单独连线的残差块组成 或者 stage中非第一个残差块的其他残差块
            blk.append(Residual(num_channels, num_channels))
    return blk

# 前两层跟goolenet一样
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 第一个stage通道数输入输出一致
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
# 第二、三、四个stage 每个stage通道数翻倍，高和宽减半
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
# 最后连接全局平均汇聚层与全连接层
net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
)

# 查看形状
X = torch.rand((1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape: ', X.shape)

# * 训练模型
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())