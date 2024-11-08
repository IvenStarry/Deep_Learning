import random
import torch
from d2l import torch as d2l

# 自然语言统计
tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])

# 最流行的词被称为停用词，对语句理解没有太多实际意义，只充当语法使用，但我们仍会在模型中去使用
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
d2l.plt.show()

# 二元语法
# corpus[:-1] 从第一个到倒数第二个词 corpus[1:] 从第二个到倒数第一个词 使用zip进行配对(1, 2)(2, 3)...
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

# 三元语法  已经出现很多有意义的文本
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

# 直观对比词元频率
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
d2l.plt.show()

# * 读取长序列模型
# ? 随机采样
# 随机生成一个小批量数据的特征和标签
def seq_data_iter_random(corpus, batch_size, num_steps): # corpus 词序列 num_steps:马尔科夫中的tao 每次取tao个词
    # 从随机偏移量开始对序列进行分区操作，随即范围包括k:num_steps - 1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 生成的子序列个数
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 随机采样，两个相邻的随机子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos): # 根据位置取序列
        return corpus[pos:pos + num_steps]

    # 可生成的batch个数
    num_batches = num_subseqs // batch_size
    # 每个取一个batchsize
    for i in range(0, batch_size * num_batches, batch_size):
        # 每个batch的起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        # 每个子序列长度为 num_steps
        X = [data(j) for j in initial_indices_per_batch]
        # 在corpus中，X偏移后一个的序列 
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

# ? 顺序分区
# 两个相邻的小批量的子序列在原始序列上也是相邻的
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

print('--------------------------------')
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

# 打包
class SeqDataLoader:  
    """加载序列数据的迭代器"""
    # use_random_iter是否随机 max_tokens若数据太大 则截取max_tokens
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

# 定义函数，返回数据迭代器和词表
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab