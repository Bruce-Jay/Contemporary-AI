import pandas as pd
from torchtext.data import Field, Example, TabularDataset, BucketIterator
import torch

# 使用pandas加载CSV文件
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# 将数字序列字符串转化为数字列表
train_data['description'] = train_data['description'].str.split(' ').apply(lambda x: [int(i) for i in x])
train_data['diagnosis'] = train_data['diagnosis'].str.split(' ').apply(lambda x: [int(i) for i in x])
test_data['description'] = test_data['description'].str.split(' ').apply(lambda x: [int(i) for i in x])

# 定义字段
SRC = Field(use_vocab=False, include_lengths=True, batch_first=True, preprocessing=torch.tensor)
TRG = Field(use_vocab=False, include_lengths=True, batch_first=True, preprocessing=torch.tensor)

fields = [('description', SRC), ('diagnosis', TRG)]

# 使用torchtext创建数据集
train_examples = [Example.fromlist([train_data.description[i], train_data.diagnosis[i]], fields) for i in range(train_data.shape[0])]
test_examples = [Example.fromlist([test_data.description[i]], fields) for i in range(test_data.shape[0])]

train_dataset = TabularDataset(path="data/train.csv", format='csv', fields=fields, examples=train_examples)
test_dataset = TabularDataset(path="data/test.csv", format='csv', fields=fields, examples=test_examples)

# 创建数据加载器
BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator = BucketIterator.splits(
    (train_dataset, test_dataset),
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x : len(x.description),
     device = device)
