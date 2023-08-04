import os
from processors.util import read_tsv

data_dir = './data_txt'
train_file = os.path.join(data_dir, 'train.tsv')
valid_file = os.path.join(data_dir, 'valid.tsv')
rows = read_tsv(valid_file)
count0 = 0
count1 = 0
count2 = 0

for i in range(1, len(rows)):
    label = rows[i][1]
    if label == '0':
        count0 += 1
    elif label == '1':
        count1 += 1
    elif label == '2':
        count2 += 1

print(count0, count1, count2)
