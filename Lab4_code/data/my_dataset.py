from data.process_data import process_data
import torch
from torch.utils.data import Dataset
from random import sample
import numpy as np
import random

class BartDataset(Dataset):
    def __init__(self, meta_data: list, max_length=160, vocab_size=1400):
        self.meta_data = meta_data
        # 序列的最大长度
        self.enc_max_length = max_length
        self.dec_max_length = max_length
        self.vocab_size = vocab_size
        # 影像医学描述
        self.des = []
        # 诊断结果
        self.diag_inputs = []
        self.dec_labels = []
        self.dec_labels_short = []
        # 特殊 tokens mask
        self.des_st_masks = []
        # attention_masks,输入的有效长度
        self.des_attention_masks = []
        self.valid_lens = []
        # 解码器 attention_masks,输入的有效长度
        self.diag_attention_masks = []
        for i in range(len(meta_data)):
            description = meta_data[i][1]

            diagnosis = meta_data[i][2]

            len1 = len(description)
            len2 = len(diagnosis)
            # description = description + [5] + diagnosis
            if len(description) > max_length:
                description = description[: max_length]
            # if len1 + len2 < self.enc_max_length:
            #     self.des_st_masks.append([0] * len1 + [1] + [0] * len2 + [1] *
            #                              (self.enc_max_length - len1 - len2 - 1))
            # else:
            #     self.des_st_masks.append([0] * len1 + [1] + [0] *
            #                              (self.enc_max_length - len1 - 1))

            self.des.append(description + [1] + [0] * (self.enc_max_length - len(description) - 2) + [2])

            self.des_attention_masks.append([1] * len(description) + [0] *
                                            (self.enc_max_length - len(description)))
            self.diag_inputs.append([2] + diagnosis + [1] + [0] * (self.dec_max_length - len(diagnosis) - 2))
            self.dec_labels.append(diagnosis + [1] + [0] * (self.dec_max_length - len(diagnosis) - 2) + [2])
            self.dec_labels_short.append(diagnosis)
            self.diag_attention_masks.append([1] * (len(diagnosis) + 1) + [0] *
                                             (self.dec_max_length - len(diagnosis) - 1))
            self.valid_lens.append(len(description))

    def __len__(self):
        return len(self.des)

    def __getitem__(self, index):
        enc_inputs = torch.tensor(self.des[index], dtype=torch.long)
        enc_attention_mask = torch.tensor(self.des_attention_masks[index], dtype=torch.long)
        dec_inputs = torch.tensor(self.diag_inputs[index], dtype=torch.long)
        dec_labels = torch.tensor(self.dec_labels[index], dtype=torch.long)
        dec_labels_short = torch.tensor(self.dec_labels_short[index], dtype=torch.long)
        dec_masks = torch.tensor(self.diag_attention_masks[index], dtype=torch.long)
        return enc_inputs, enc_attention_mask, dec_inputs, dec_labels, dec_masks


class PredDataset(Dataset):
    def __init__(self, meta_data:list, encode_max_length=160, decode_max_length=96, vocab_size=1400):
        self.meta_data = meta_data
        self.enc_max_length = encode_max_length
        self.ids = []
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        for i in range(len(meta_data)):
            id = [int(meta_data[i][0])]
            description = meta_data[i][1]
            self.ids.append(id)
            self.input_ids.append(description + [0] * (self.enc_max_length - len(description)))
            self.attention_masks.append([1] * len(description) + [0] * (self.enc_max_length - len(description)))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        ids = torch.tensor(self.ids[index], dtype=torch.long)
        input_ids = torch.tensor(self.input_ids[index], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_masks[index], dtype=torch.long)
        return ids, input_ids, attention_mask

class BartDatasetWord(Dataset):
    def __init__(self, meta_data: list, max_length=160, vocab_size=1400):
        self.meta_data = meta_data
        # 序列的最大长度
        self.enc_max_length = max_length
        self.dec_max_length = max_length
        self.vocab_size = vocab_size
        # 影像医学描述
        self.des = []
        # 诊断结果
        self.diag_inputs = []
        self.dec_labels = []
        self.dec_labels_short = []
        # 特殊 tokens mask
        self.des_st_masks = []
        # attention_masks,输入的有效长度
        self.des_attention_masks = []
        self.valid_lens = []
        # 解码器 attention_masks,输入的有效长度
        self.diag_attention_masks = []
        for i in range(len(meta_data)):
            description = meta_data[i][1]

            diagnosis = meta_data[i][2]

            len1 = len(description)
            len2 = len(diagnosis)
            # description = description + [5] + diagnosis
            if len(description) > max_length:
                description = description[: max_length]

            self.des.append(description + '1' + '0' * (self.enc_max_length - len(description) - 2) + '2')

            self.des_attention_masks.append('1' * len(description) + '0' *
                                            (self.enc_max_length - len(description)))
            self.diag_inputs.append('2' + diagnosis + '1' + '0' * (self.dec_max_length - len(diagnosis) - 2))
            self.dec_labels.append(diagnosis + '1' + '0' * (self.dec_max_length - len(diagnosis) - 2) + '2')
            self.dec_labels_short.append(diagnosis)
            self.diag_attention_masks.append('1' * (len(diagnosis) + 1) + '0' *
                                             (self.dec_max_length - len(diagnosis) - 1))
            self.valid_lens.append(len(description))

    def __len__(self):
        return len(self.des)

    def __getitem__(self, index):
        enc_inputs = torch.tensor([ord(c) for s in self.des[index] for c in s])
        enc_attention_mask = torch.tensor([ord(c) for s in self.des_attention_masks[index] for c in s])
        dec_inputs = torch.tensor([ord(c) for s in self.diag_inputs[index] for c in s])
        dec_labels = torch.tensor([ord(c) for s in self.dec_labels[index] for c in s])
        dec_masks = torch.tensor([ord(c) for s in self.diag_attention_masks[index] for c in s])
        return enc_inputs, enc_attention_mask, dec_inputs, dec_labels, dec_masks

