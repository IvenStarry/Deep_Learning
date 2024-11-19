import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

# * 下载香蕉检测数据集
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

# * 读取数据集
#@save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    # iterrows() 逐行处理数据
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))

    '''
    torch.unsqueeze(dim) 在第dim个维度增加一个维度
    维度组成：batch_size, num_object, feature(类别和四个框坐标) e.g.[32, 1, 5]
    这里的targets是边缘框的四个坐标，除以256归一化，方便框架输入，类似ToTensor()
    增加一个维度(位置在第1维度即目标个数)表示一个图片中最多出现的物体数，这里只有香蕉所以是1
    '''
    return images, torch.tensor(targets).unsqueeze(1) / 256

# 创建自定义Dataset实例
class BananaDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read' + str(len(self.features)) +
                            (f' training examples' if is_train else f'validation examples'))
    
    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])
    
    def __len__(self):
        return len(self.features)

# 为训练集和测试集返回两个数据加载器实例
def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananaDataset(is_train=True), batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananaDataset(is_train=False), batch_size)
    return train_iter, val_iter

# 查看shape
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)

# * 演示
# permute 维度转置 通道数放在最后
# 除以255 归一化 用于显示 库的要求
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)

for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
d2l.plt.show()