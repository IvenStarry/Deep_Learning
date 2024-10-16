import torch
from d2l import torch as d2l
from torch import nn

# * VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []

    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# * VGG网络
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) # 定义模型卷积层个数和输出的通道数
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    return nn.Sequential(
        *conv_blks, 
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), # 池化5次 224 -> 7
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )
net = vgg(conv_arch)

# 观察输出形状
X = torch.randn((1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'Output shape:\t', X.shape)

# * 训练
# 因为vgg-11比alexnet计算量更大，因此构建一个通道数少的网络
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

# 正式训练
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())