import os

import torch
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers import GPT2Tokenizer, TFGPT2Model
from data.process_data import process_data

if torch.cuda.is_available() is False:
    raise EnvironmentError("not find GPU device for training")
device = torch.device('cuda')


data = process_data('./', 'data/train.csv', mode='train')
test_data = process_data('./', 'data/test.csv', mode='test')

# 删除模型缓存
cache_dir = "./cache"
if os.path.exists(cache_dir):
    os.rmdir(cache_dir)

total_data = data + test_data
print(total_data[0][1])
input_ids = total_data[0][1]
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
result = tokenizer.convert_ids_to_tokens(input_ids)
print(result)