import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# * 简单网络(LeNet改)
# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# * 数据同步
# 向多个设备分发参数
def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)

# allreduce 将所有向量相加，并将结果广播给所有GPU
def allreduce(data): # 把所有数据放在 GPU:0 上
    for i in range(1, len(data)): # for loop GPU 0 以外的GPU
        data[0][:] += data[i].to(data[0].device) # 把第i个GPU上的data传给第0个tensor的GPU上，再加到 GPU0 对应的数据上
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device) # 将新的结果复制到第i个GPU上

data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i+ 1) for i in range(2)]
print('before allreduce: \n', data[0], '\n', data[1])
allreduce(data)
print('after allresuce: \n', data[0], '\n', data[1])

# * 数据分发
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices) # 根据GPU的个数，将data均匀地切开
print('input', data)
print('load into', devices)
print('output', split)

def split_batch(X, y, devices): # 方便使用，集成起来
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices), nn.parallel.scatter(y, devices))

# * 训练
# 在小批量上实现多GPU训练
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    ls = [loss(lenet(X_shard, device_W), y_shard).sum() for X_shard, y_shard, device_W in zip(X_shards, y_shards, device_params)]
    for l in ls:
        l.backward()
    with torch.no_grad():
        for i in range(len(device_params[0])): # i 层数
            # 对每一个每一个层
            allreduce([device_params[c][i].grad for c in range(len(devices))]) # c GPU个数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 因为每个GPU上参数一致，梯度经过上面的操作也一致，所有在每个GPU上做参数更新

# 定义训练函数
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    device_params = [get_params(params, d) for d in devices] # 复制参数到每一个GPU上

    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 为单个小批量执行多GPU训练
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize() # 同步一次 保证每一个GPU都运算完成
        timer.stop()
        # 在GPU0上评估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
            f'在{str(devices)}')

# 在单GPU上运行
train(num_gpus=1, batch_size=256, lr=0.2)
# 在双GPU上运行
train(num_gpus=2, batch_size=256, lr=0.2)