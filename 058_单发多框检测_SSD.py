import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# * 模型
# 类别预测层 预测锚框类别
# 目标是n类，锚框总共有n+1个类别，其中0类是背景
def cls_predictor(num_inputs, num_anchors, num_classes): # 输入图像输入尺寸，锚框个数，目标类别个数
    # 返回num_anchors * (num_classes + 1) 对应每一个锚框对每一种类别的预测值
    # 此卷积层的输入和输出的宽度和高度保持不变 类似于卷积层只是参数更加少
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

# 边界框预测层 预测锚框与真实框的偏移
def bbox_predictor(num_inputs, num_anchors):
    # 返回num_anchors * 4 给每个锚框预测4个偏移量
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# 连接多尺度的预测
def forward(x, block):
    return block(x)

# 生成2个特征图 高宽是20 每个中心点生成5个锚框 类别是10
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape, Y2.shape)
# (torch.Size([2, 55, 20, 20]) 第一个尺度每一个像素点生成55个预测值 11*5 总共20*20个像素点
# torch.Size([2, 33, 10, 10])) 第二个尺度每一个像素点生成335个预测值 11*3 总共10*10个像素点

# 连接除了批量大小以外的所有参数
def flatten_pred(pred):
    # 先把通道数放在最后，把每个像素的每个锚框预测的类别放在一起
    # 从第一个维度开始执行展平操作，shape变为（批量大小，高*宽*通道数）
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

# 连接不同尺度下的框
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# 尽管Y1和Y2在通道数、高度和宽度方面具有不同的大小，我们仍然可以在同一个小批量的两个不同尺度上连接这两个预测输出
print(concat_preds([Y1, Y2]).shape)

# 高和宽减半块 下采样块 变换通道数
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)

# 基本网络块
def base_net():
    blk = []
    # 通道数从3开始，再到16，再到32，再到64 同时高宽也减半三次
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)
print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)

# 完整的模型 五个模块构成
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        # 将高度和宽度压到1
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

# 给每一个块定义前向计算
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    # 生成当前fmap的卷积层输出
    Y = blk(X)
    # 生成当前fmap的锚框
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    # 生成类别和偏移预测
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

# 超参数
# 最下层就是fmap尺寸较大的用小的锚框，后面锚框尺寸越来越大
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        # 每个Stage输出通道数
        idx_to_in_channels = [64, 128, 128, 128, 128]
        # 做五次预测
        for i in range(5):
            # setattr 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        # 把类别作为最后一维拿出来 方便预测
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape) # 所有stage每个像素每个anchor之和
print('output class preds:', cls_preds.shape) # 32 batchsize 2 类别加1 
print('output bbox preds:', bbox_preds.shape) # anchor*4 每个锚框与真实框的偏移

# * 训练模型
# 读取数据集
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

# 初始化参数
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 定义损失函数和评价函数
cls_loss = nn.CrossEntropyLoss(reduction='none') # 类别损失
bbox_loss = nn.L1Loss(reduction='none') # 偏移损失 选择L1的原因：如果预测与真实差距特别大，不会返回一个特别大的损失

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1) # mask标注了背景框和非背景框
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# 正式训练
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

# 保存参数
torch.save(net.state_dict(), 'related_data/SSD.pth')
print("Model parameters saved to SSD.pth")

# * 预测目标
X = torchvision.io.read_image('related_data/banana.jpeg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

# Load model parameters before testing
net_load = TinySSD(num_classes=1).to(device)  # Make sure to initialize the model first
net_load.load_state_dict(torch.load('related_data/SSD.pth'))
print("Model parameters loaded from SSD.pth")

def predict(X):
    net_load.eval()
    anchors, cls_preds, bbox_preds = net_load(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)

def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    d2l.plt.show()

display(img, output.cpu(), threshold=0.9)