import hashlib
import os
import tarfile
import zipfile
import requests

# * 下载和缓存数据集
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('.', 'related_data')):
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
            sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname # 命中缓存
    print(f'正在从{url}中下载{fname}')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

# * 访问和读取数据集
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# 查看数据集大小
print(train_data.shape)
print(test_data.shape)

# 查看前四个和最后两个特征，以及标签
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 训练不需要用到特征id，训练数据最后一列是label，故删除 concat连接多个数组 
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) # [1, -1]的作用是去除前面1列和最后一列
print(all_features)

'''
a[1:-1]
a[n:-n]作用是去除前n个元素和末n个元素
>>> a=(1,2,3,4,5)
>>> a[1:-1]
(2, 3, 4)
>>> a[2:-2]
(3,)

a[-1]
a[-n]作用是取倒数第n个元素
>>> a=(1,2,3,4,5)
>>> a[-2]
4

a[:-1]
a[:-n]的作用是去除后n个元素

>>> a=(1,2,3,4,5)
>>> a[:-1]
(1, 2, 3, 4)

a[::-1]
a[::-1]的作用是将所有元素逆序排列
>>> a[::-1]
(5, 4, 3, 2, 1)

a[n::-1] 的作用是从第n个元素截取后逆序排列
>>> a=(1,2,3,4,5)
>>> a[2::-1]
(3, 2, 1)
'''

# * 数据预处理
# 因为每一列的数据有大有小，因此将数值列缩放到均值为0，方差为1来标准化数据， x <- (x - miu)/sigma
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index # object是离散类型()数据,即pandas里的object是python中的str
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理离散值 使用独热编码(one-hot)  get_dummies()将被类别变量转独热 dummy_na是否包括表示缺失值的列
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

# 从pandas格式中提取Numpy格式，并转换为张量表示
n_train = train_data.shape[0] # 获取训练集样本个数
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# * 训练
loss = nn.MSELoss()
in_features = train_features.shape[1] # 特征个数

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

# 预测值应关注相对误差(y-y_hat)/y，即偏离的百分比，而非绝对误差y-y_hat，即偏离的具体大小(相同的绝对误差，在房价原本值比较小时模型较差，房价原本值较大时模型较好)
def log_rmse(net, features, labels): # 取log规避除法
    # torch.clamp(x, a, b) 将x截断在[a,b]内 小于a则置为a 大于b则置为b
    clipped_preds = torch.clamp(net(features), 1, float('inf')) # 为了稳定log取值
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
            num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    
    return train_ls, test_ls # 损失

# * K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k # 这里 \\ 表示算出除数并向下取整
    X_train, y_train = None, None
    for j in range(k):
        # slice(start,end)：方法可从已有数组中返回选定的元素，返回一个新数组，包含从start到end（不包含该元素）的数组元素
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i: # 第几折是验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None: # 如果第一次遇到训练集部分
            X_train, y_train = X_part, y_part
        else:
            # torch.cat() 将两个张量按指定维度拼接
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    
    return X_train, y_train, X_valid, y_valid

# 返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size) # 这里*的作用是构造与解构，data传回四个tuple,*data就是把data拆开，还原为4个
        # 取每一折最后一个epoch的loss，在训练过程的最后一个epoch。模型通常已经收敛，参数基本稳定
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                    xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                    legend=['train', 'valid'], yscale='log')
            d2l.plt.show
        
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f},'
                f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

# * 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
        f'平均验证log rmse: {float(valid_l):f}')

# * 提交Kaggle预测
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse: {float(train_ls[-1]):f}')

    # 应用于测试集
    preds = net(test_features).detach().numpy()

    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./related_data/submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
                num_epochs, lr, weight_decay, batch_size)