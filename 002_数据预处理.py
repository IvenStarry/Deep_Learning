import os
# os.markdirs()创建目录 exist_ok设置为True当目录存在时不会报错
os.makedirs(os.path.join('.', 'related_data'), exist_ok=True)
# 创建一个人工数据集，并存储在CSV文件(逗号分隔值)
data_file = os.path.join('.', 'related_data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRoom, Alley, Price\n') # 列明
    f.write('NA, Pave, 127500\n') # 每行表示一个数据样本
    f.write('2, NA, 106000\n')
    f.write('4, NA, 178100\n')
    f.write('NA, NA, 140000')

# pandas 从创建的csv文件中加载原始数据集
import pandas as pd
data = pd.read_csv(data_file)
print(data)

# 处理缺失的数据
# * 方法1：插值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 对于数值列：fillna() 对所有是na的域赋值 mean中的numeric_only设置为True则只处理数值列的均值
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)
# 对于非数值列(类别/离散)：将“NAN”视为一个类别
# pd.get_dummies 将类别变量（categorical variables）转换为独热编码 dummy_na: 是否包括表示缺失值的列
inputs = pd.get_dummies(inputs, dummy_na=True).astype(int)
print(inputs)
# 方法2：删除

import torch
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)
'''
view()与reshape()区别：(老版本)
- view()和reshape()均用于调整tensor的形状。
- reshape()不会检查张量的连续性，因此它可能在某些情况下失败或产生意外的结果，可以使用-1自动推断某些维度。
- 而view()会检查张量的连续性，并在不满足条件时抛出错误，从而提供了更高的安全性，不可以使用-1自动推断某些维度

最新版 PyTorch 2.4.1 view和reshape暂无发现区别
'''