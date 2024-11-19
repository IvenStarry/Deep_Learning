import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# * 构造模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
# 查看后三层的网络结构
print(list(pretrained_net.children())[-3:])

# 创建一个全卷积网络实例net
net = nn.Sequential(*list(pretrained_net.children())[:-2]) # 去掉全局池化层和全连接层
X = torch.rand(size=(1, 3, 320, 480))
print(net(X).shape)

num_classes = 21 # pascal voc2012数据集类别数
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                    kernel_size=64, padding=16, stride=32))

# * 初始化转置卷积层
# 使用经过双线性插值初始化的核，再使用转置卷积层，可以实现双线性插值的上采样
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

# 实验
# 输出图像的高宽翻倍
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
img = torchvision.transforms.ToTensor()(d2l.Image.open('related_data/cat_dog.png').convert('RGB'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0))
d2l.plt.show()
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img)
d2l.plt.show()

# 用双线性插值的上采样初始化转置卷积层，对于卷积层使用Xavier初始化参数
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# * 读取数据集
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

# * 训练
def loss(inputs, targets):
    # 计算损失的结果是三维矩阵(样本，高，宽) 计算高和宽的矩阵就可以得到一个一维的值(每个样本的损失)
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# * 预测
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    # argmax(dim=1) 在每一个像素有21个通道(类别) 找个通道上最大的值即预测类别
    pred = net(X.to(devices[0])).argmax(dim=1)
    # reshape成跟输入图像尺寸一样
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
d2l.plt.show()