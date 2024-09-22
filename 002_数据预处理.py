import os
import numpy as np
# os.markdirs()创建目录 exist_ok设置为True当目录存在时不会报错
os.makedirs(os.path.join('.', 'related_data'), exist_ok=True)
# 创建一个人工数据集，并存储在CSV文件(逗号分隔值)
data_file = os.path.join('.', 'related_data', 'house_tiny.csv')
# ! 为了识别缺失值NaN, NA前面不能有空格
with open(data_file, 'w') as f:
    f.write('NumRoom,Alley,Price\n') # 列明
    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000')

# pandas 从创建的csv文件中加载原始数据集
import pandas as pd
data = pd.read_csv(data_file)
print(data)
print(type(data))

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

import torch
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)

# * 方法2：删除缺失值最多的列
# 实现1
count = 0
count_max = 0
col_names = data.columns.values # .values返回一个ndarray数组
for col_name in col_names:
    count = data[col_name].isna().sum() # isna() 检测缺失值
    print(f"column name:{col_name}, NAN numbers:{count}")
    if count > count_max:
        count_max = count
        drop_col_name = col_name

data_drop = data.drop(drop_col_name, axis=1)
print(data_drop)

# 实现2
NAN_count = data.isna().sum() # 返回每一列的缺失值的总数
print(NAN_count)
label = NAN_count.idxmax() # idxmax返回最大值的索引 argmax返回最大值的下标 max返回最大值
print(label)
new_data = data.drop(label, axis=1)
print(new_data)