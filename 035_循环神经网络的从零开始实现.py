import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# * 独热编码
# 长度是28，将下标为0或2的数值置1，其余置0, shape:(tensor, num_classes)
print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

# 假定一个小批量 (批量大小， 时间步数)
X = torch.arange(10).reshape((2, 5))
# 转置目的x[0,:,:]表示T(0)状态,X[1,:,:]表示T(1)状态，满足时序关系，实现并行计算
print(F.one_hot(X.T, 28).shape)

# * 初始化循环神经网络模型的模型参数
def get_params(vocab_size, num_hiddens, device):
    # 输入维度是oneshot的尺寸也就是bocab_size，输出就是对下一个单词的预测，也等于vocab_size
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # 均值为0 方差为0.01
        return torch.randn(size=shape, device=device) * 0.01
    
    # 隐藏层的参数
    # 对输入映射到隐藏层的矩阵 
    W_xh = normal((num_inputs, num_hiddens))
    # 上一时刻的隐藏变量到下一时刻的隐藏变量
    W_hh = normal((num_hiddens, num_hiddens))
    # 到隐藏层的偏置项
    b_h = torch.zeros(num_hiddens, device=device)

    # 输出层参数
    # 隐藏层映射到输出的矩阵
    W_hq = normal((num_hiddens, num_outputs))
    # 到输出的偏置项
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# * 循环神经网络模型
# 在初始化时返回隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# state 初始化的隐藏状态 params可学习的参数
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # input shape(time_step, batch_size, vocab_size) 这里for取出的一个step的矩阵
    for X in inputs:
        # H当前时刻的隐藏元 X, W_xh (batch_size, num_hiddens)  H, W_hh(batch_size, num_hiddens) b_h(num_hiddens)
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # Y当前时刻的输出
        Y = torch.mm(H, W_hq) + b_q
        # output shape(time_step, batch_size, vocab_size)
        outputs.append(Y)
    # 拼接后变成了(time_step*batch_size, vocab_size)
    return torch.cat(outputs, dim=0), (H,)

# 包装成类
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
    
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

# 例子
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
# Y(bz*ts, vs) new_state[0] (bs, num_hiddens)
print(Y.shape, len(new_state), new_state[0].shape)

# * 预测
def predict_ch8(prefix, num_preds, net, vocab, device): # prefix开头提示句 num_preds预测字个数
    state = net.begin_state(batch_size=1, device=device)
    # 拿到开头词的下标放在outputs
    outputs = [vocab[prefix[0]]]
    # 将output最后一个词作为下一个的输入 time_step:1, batch_size:1
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        # 根据训练好的模型权重得到这一时刻的状态H
        _, state = net(get_input(), state)
        # 把此时真实的输出存入outputs
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        # 输入是output中最后一个字母
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1))) # 第一维度就是vocab_size(独热)预测最大值的下标，reshape成标量
    return ''.join([vocab.idx_to_token[i] for i in outputs])

predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())

# * 梯度裁剪
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    
    # L2norm 正则化
    # params是所有层的梯度，p是每一层的梯度，先求每一层的sum，再求全局sum
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# * 训练
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时(因为时序信息不连续)初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量 断开前面的链式求导 只关心现在以及后面的状态
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        # 真实输出拉成向量 配合y_hat
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        # y_hat (time_step*batch_size, vocab_size)    .long()将数字或者字符串转为长整型
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            # 梯度裁剪
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    # math.exp(metric[0] / metric[1]) 困惑度 metric[0]：loss累加 metric[1]：样本总数
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

# 正式训练
num_epochs, lr = 500, 1
# 顺序
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
# 随机
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)