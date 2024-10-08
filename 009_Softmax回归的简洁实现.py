import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# * 初始化模型参数
# Softmax回归的输出层是一个全连接层
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m): # m是Layer层
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01) # 初始化

net.apply(init_weights)

# * 重新审视Softmax的实现
# 在交叉熵损失函数中传递未归一化的预测，并同时计算Softmax及其对数
loss = nn.CrossEntropyLoss(reduction='none') 

# * 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# * 训练
# 调用之前定义的训练函数来训练模型
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()