import collections
import re
from d2l import torch as d2l

# * 读取数据集
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 将除了字母以外的所有字符(标点符号，不认识的字母)全部变成空格 简化数据
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

# * 词元化
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word': # 拆分为单词
        return [line.split() for line in lines]
    elif token == 'char': # 拆分为字母
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines, token='char')
for i in range(1):
    print(tokens[i])

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])

# * 词表
# 构建一个字典，将字符串类型的标记映射到从0开始的数字索引中
class Vocab: 
    """文本词表"""
    # min_freq 若一个单词少于这个值则舍弃 reserved_tokens表示句子开始或者句子结束的token
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率从大到小排序token   key:排序的比较  .items()字典转元组 x[1]即词频，元组的第二个元素 
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0 '<unk>'用来表示词汇表中没有的单词
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 字典推导式 将token和id对应起来
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        # 添加高频词
        for token, freq in self._token_freqs:
            if freq < min_freq: # 小于最小出现频率 则舍弃
                break
            if token not in self.token_to_idx: # 检查是否已经存在于词汇表
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1 # 记录该词的索引位置
    
    # 返回token个数
    def __len__(self):
        return len(self.idx_to_token)

    # 给token返回下标
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            # get用于取出字典指定key的value，若没有key默认值是self.unk
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens] # 一直拆分成直至不是list或tuple类型(str)

    # 给下标返回tokens
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        '''
        外层迭代：for line in tokens，表示遍历 tokens 的每一行（即每个子列表）。
                在我们的示例中，这将依次取出 ['I', 'love', 'coding'] 和 ['Python', 'is', 'great']。
        内层迭代：对于每一个 line，再使用 for token in line 遍历每个子列表 line 中的单词（或“token”）。
        生成新列表：对于每一个 token，将其添加到新的列表中。
        '''
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])

# * 整合所有功能
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# corpus是每一个的字符下标 vocab是字表 28是因为实例选择了char参数，返回26个字母+<unk>+空格=28
corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))