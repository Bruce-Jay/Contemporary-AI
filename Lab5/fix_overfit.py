import os
from processors.util import read_tsv
import csv
import random

data_dir = './data_txt'
train_file = os.path.join(data_dir, 'train.tsv')
valid_file = os.path.join(data_dir, 'valid.tsv')

def fix_unbalanced(filepath):
    rows = read_tsv(filepath)

    count = len(rows)

    for i in range(1, len(rows)):

        if rows[i][1] == '1':
            # 如果下标为 1 的值等于 1，'a' 模式就将该行追加到 tsv 文件夹的末尾
            with open(filepath, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                label = rows[i][1]
                imgid = rows[i][2]
                text = rows[i][3]
                f.write('%d\t%s\t%s\t%s\n' % (count, label, imgid, text))
                count += 1
                f.write('%d\t%s\t%s\t%s\n' % (count, label, imgid, text))
                count += 1

    print('task finished.')


def shuffle(filepath):
    # remove the head
    rows = read_tsv(filepath)[1:]
    random.shuffle(rows)
    count = 1
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        f.write('index\ttag\tguid\tdescription\n')
        for i in range(1, len(rows)):
            label = rows[i][1]
            imgid = rows[i][2]
            text = rows[i][3]
            f.write('%d\t%s\t%s\t%s\n' % (count, label, imgid, text))
            count += 1

fix_unbalanced(train_file)
shuffle(train_file)
