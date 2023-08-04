import os

labelMap = {"negative": 0, "neutral": 1, "positive": 2}

data_dir = './data_txt'
train_file = os.path.join(data_dir, 'train.txt')
test_file = os.path.join(data_dir, 'test_without_label.txt')

file_raw = open(train_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
lines = file_raw.readlines()

# 生成训练集数据
with open(os.path.join(data_dir, 'train.tsv'), 'w', encoding='utf-8') as f:
    f.write('index\ttag\tguid\tdescription\n')
    count = 0
    # 希望按照8：2的比例划分训练集和验证集
    for i in range(1, 3200):
        count += 1
        # 去除首尾空格
        line = lines[i].strip()
        guid, tag = line.split(',')
        file1 = './data/' + guid + '.txt'
        with open(file1, 'r', encoding='gb18030') as file_text:
            lines_text = file_text.readlines()
            text = ''
            for index in range(len(lines_text)):
                text += lines_text[index].strip() + '\t'
            polarity = labelMap[tag]
            imgid = guid
            label = str(int(polarity))
            f.write('%d\t%s\t%s\t%s\n' % (count, label, imgid, text))

#  验证集
with open(os.path.join(data_dir, 'valid.tsv'), 'w', encoding='utf-8') as val_f:
    val_f.write('index\ttag\tguid\tdescription\n')
    count = 0
    for i in range(3200, len(lines)):
        count += 1
        # 去除首尾空格
        line = lines[i].strip()
        guid, tag = line.split(',')
        file1 = './data/' + guid + '.txt'
        with open(file1, 'r', encoding='gb18030') as file_text:
            lines_text = file_text.readlines()
            text = ''
            for index in range(len(lines_text)):
                text += lines_text[index].strip() + '\t'
            polarity = labelMap[tag]
            imgid = guid
            label = str(int(polarity))
            val_f.write('%d\t%s\t%s\t%s\n' % (count, label, imgid, text))

# 测试集
file_raw_test = open(test_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
lines_test = file_raw_test.readlines()
with open(os.path.join(data_dir, 'test.tsv'), 'w', encoding='utf-8') as test_f:
    test_f.write('index\ttag\tguid\tdescription\n')
    count = 0
    for i in range(1, len(lines_test)):
        count += 1
        guid = lines_test[i].strip()
        guid, tag = guid.split(',')
        file1 = './data/' + guid + '.txt'
        with open(file1, 'r', encoding='gb18030') as file_text:
            lines_text = file_text.readlines()
            text = ''
            for index in range(len(lines_text)):
                text += lines_text[index].strip() + '\t'
            imgid = guid
            label = tag
            test_f.write('%d\t%s\t%s\t%s\n' % (count, label, imgid, text))

print('Data gotten!')


