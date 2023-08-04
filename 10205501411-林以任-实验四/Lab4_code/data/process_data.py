import os
# 导入正则表达式模块，替换数据集中可能出现的空格
import re
import pandas as pd
import sentencepiece as spm
import csv

spm_model = spm.SentencePieceProcessor()
spm_model.load('m.model')


def process_data(root:str, file_path:str, mode='train'):
    file = os.path.join(root, file_path)
    df = pd.read_csv(file, header=None, encoding='utf-8')

    len_data = 0
    data = []
    my_dict = set()
    data_str = ''
    res = []
    for i in range(1, len(df)):
        lst = [df[0][i]]
        # 移除开头和结尾的空格或者换行，并且将字符串中的多个空格转换为一个
        description = re.sub(' +', ' ', df[1][i].strip())
        res.append(description)
        description = [int(x) for x in description.split()]
        for i in range(len(description)):
            my_dict.add(description[i])
        lst.append(description)
        len_data = max(len_data, len(description))
        if mode == 'train':
            # 如果模式定义是训练模式，则需要读取 diagnosis
            diagnosis = re.sub(' +', ' ', df[2][i].strip())
            res.append(diagnosis)
            diagnosis = [int(x) for x in diagnosis.split()]
            lst.append(diagnosis)
            for i in range(len(diagnosis)):
                my_dict.add(diagnosis[i])
        data.append(lst)
    print(mode + '最大的 description 数据长度为：', len_data)
    print(mode + '的字典集合长度为：', len(my_dict))
    return data


def process_data_str(root:str, file_path:str, mode='train'):
    file = os.path.join(root, file_path)
    df = pd.read_csv(file, header=None, encoding='utf-8')

    len_data = 0
    data = []
    my_dict = set()
    data_str = ''
    res = []
    for i in range(len(df)):
        lst = [df[0][i]]
        # 移除开头和结尾的空格或者换行，并且将字符串中的多个空格转换为一个
        description = re.sub(' +', ' ', df[1][i].strip())
        res.append(description)
        description_list = []

        # description = [int(x) for x in description.split()]
        for i in range(len(description)):
            my_dict.add(description[i])
        lst.append(description)
        len_data = max(len_data, len(description))
        if mode == 'train':
            # 如果模式定义是训练模式，则需要读取 diagnosis
            diagnosis = re.sub(' +', ' ', df[2][i].strip())
            res.append(diagnosis)
            diagnosis_list = []

            lst.append(diagnosis)
            for i in range(len(diagnosis)):
                my_dict.add(diagnosis[i])
        data.append(lst)
    print(mode + '最大的 description 数据长度为：', len_data)
    print(mode + '的字典集合长度为：', len(my_dict))
    return data


def transfer_numbers_to_words_train(root:str, file_path:str):
    file = os.path.join(root, file_path)
    df = pd.read_csv(file, header=None, encoding='utf-8')
    with open('data/words_train.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)

        for i in range(1, len(df)):
            lst = df[0][i]
            lst = int(lst)
            # print(lst)
            # 移除开头和结尾的空格或者换行，并且将字符串中的多个空格转换为一个
            description = re.sub(' +', ' ', df[1][i].strip())
            description = [int(x) for x in description.split()]
            # print(description)
            words_description = spm_model.decode(description)
            # words_description = ' '.join(words_description)

            # 如果模式定义是训练模式，则需要读取 diagnosis
            diagnosis = re.sub(' +', ' ', df[2][i].strip())
            diagnosis = [int(x) for x in diagnosis.split()]
            words_diagnosis = spm_model.decode(diagnosis)

            # words_diagnosis = ' '.join(words_diagnosis)
            # print(words_diagnosis)
            writer.writerow([lst, words_description, words_diagnosis])


def transfer_numbers_to_words_test(root:str, file_path:str):
    file = os.path.join(root, file_path)
    df = pd.read_csv(file, header=None, encoding='utf-8')
    with open('data/words_test.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)

        for i in range(1, len(df)):
            lst = df[0][i]
            lst = int(lst)
            # 移除开头和结尾的空格或者换行，并且将字符串中的多个空格转换为一个
            description = re.sub(' +', ' ', df[1][i].strip())
            description = [int(x) for x in description.split()]
            words_description = spm_model.decode(description)
            # words_description = ' '.join(words_description)

            writer.writerow([lst, words_description])


