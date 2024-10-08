import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# * 初始化模型参数
# 对于Softmax回归，输入是一个向量，因此将展平每个图像，将它们视为长度为784的向量
num_inputs = 784
num_outputs = 10 # 10个类别

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# * 定义Softmax操作
# 给定一个矩阵X，我们可以对所有元素求和
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

# 实现Softmax
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True) # 对行进行求和(每个样本是一行，上面将图像展平)
    # print(partition)

    return X_exp / partition # 利用广播机制

# 将没每个元素变成一个非负数，根据概率原理，每行总和为1
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))

# * 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b) # W.shape[0]=784 -1这里计算的是batch_size * dims

# * 定义损失函数
# 创建数据y_hat，其中包含2个样本在3个类别的预测概率，使用y作为y_hat中概率的索引
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y]) # 索引数组 第一个子数组选择行 第二个子数组选择列[[0, 1], [0, 2]] 输出[0,0]和[1,2]的数据

# 实现交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y]) # len计算第0维度
print(cross_entropy(y_hat, y))

# 将预测类别与真实y元素进行比较
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # 判断张量是否大于大于1维度，判断列(第二个维度是否大于1)
        y_hat = y_hat.argmax(axis=1)
    # 由于==对数据类型比较敏感，为了确保y_hat y的数据类型一致，将y_hat转为y的数据类型，结果是一个包含0(错)与1(对)的张量
    cmp = y_hat.type(y.dtype) == y 

    return float(cmp.type(y.dtype).sum()) # 将cmp类型转为tensor并求和，得到正确预测的数量
# 计算正确率
print(accuracy(y_hat, y) / len(y))

# * 分类精度
# 评估在任意模型net的准确率
def evaluate_accuracy(net, data_iter):
    '''
    isinstance()用来判断一个对象是否是一个已知的类型 
    isinstance(object,classtype)
    object -- 实例对象。
    classtype -- 可以是直接或间接类名、基本类型或者由它们组成的元组。
    '''
    if isinstance(net, torch.nn.Module):
        net.eval() # 转评估模型
    
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    
    return metric[0] / metric[1]

# 示例创建2个变量，用于存储正确预测和预测的总数量
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx): # 使对象可以使用索引操作
        return self.data[idx]

'''
CPU性能不够，可能会导致发生DataLoader worker exited unexpectedly

原因：采用多进程加载数据时，python需要重新导入模块以启动每个子进程，这个重复过程会执行模块中的顶级代码，若CPU性能不足以多次调用模块，则会报错

解决方法：
1. 在d2l包中的torch模块get_dataloader_workers()返回值改为0 不启用多线程 这里采用方法1
2. 将下面代码放入if __name__ == '__main__'，可以防止模块被多次导入时重复执行evaluate_accuracy函数

'''
print(evaluate_accuracy(net, test_iter))

# * 训练
# Softmax回归的训练
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    
    metric = Accumulator(3)

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer): # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            # 这里乘len(y)的原因是 pytorch自动对loss取均值
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size.numel()) # loss累加值 正确个数 样本总个数
        
        else: # 使用自己写的优化器和损失函数
            l.sum().backward() # 自己写的损失是一个向量
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    
    return metric[0] / metric[2], metric[1] / metric[2] # 返回训练loss和正确率acc

# 定义一个在动画中绘制数据的实用程序类 (暂时跳过)
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        d2l.plt.pause(0.01)
        display.clear_output(wait=True)

# 训练函数
def train_ch3(net,train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater) # 训练损失
        test_acc = evaluate_accuracy(net, train_iter) # 测试集精度
        animator.add(epoch + 1, train_metrics + (test_acc, ))
    
    train_loss, train_acc = train_metrics

lr = 0.1
# 使用小批量随机梯度下降来优化模型的损失函数
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

# 训练模型10个迭代周期
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
d2l.plt.show()

# * 预测
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break

    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()

predict_ch3(net, test_iter)