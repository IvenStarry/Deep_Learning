import math
import torch
from torch import nn
from d2l import torch as d2l

# * 掩蔽softmax操作
# 只保留有意义的词元
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一个轴上被屏蔽的元素使用一个特别大的负值替换，使得softmax输出为0 (e^x)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 演示
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

# * 加性注意力
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # k 到 h  k长度转换
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        # q 到 h  q长度转换
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        # v 到 1
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens):
        # queries 的形状变为 (batch_size, num_queries, num_hiddens)
        # keys 的形状变为 (batch_size, num_keys, num_hiddens)
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 增加维度方便进行广播
        # queries.unsqueeze(2) (batch_size, num_queries, 1, num_hiddens)。
        # keys.unsqueeze(1) (batch_size, 1, num_keys, num_hiddens)。
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # features (batch_size, num_queries, num_keys, num_hiddens)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。(batch_size, num_queries, num_keys, 1)
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数) (batch_size, num_queries, num_keys)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        # dropout随机将一部分注意力权重置0防止过拟合，依赖某些特定的注意力
        return torch.bmm(self.dropout(self.attention_weights), values)

# 演示
# 查询 键值shape（批量大小，步数或词元序列长度，特征大小）
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
print(attention(queries, keys, values, valid_lens))

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.show()

# * 缩放点积注意力
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
print(attention(queries, keys, values, valid_lens))

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.show()