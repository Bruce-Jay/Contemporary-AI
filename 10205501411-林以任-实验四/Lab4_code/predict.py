import argparse
import os
from tqdm import tqdm

from data.process_data import process_data
from data.dataloader import pred_dataloader
from model.small_bart import Seq2SeqModel
from opts import *
import pandas as pd

def decode(model, test_loader, args, device, data_desc):
    model.eval()
    num_steps = len(test_loader)
    # 在进程 0 中打印训练进度
    data_loader = tqdm(test_loader, total=num_steps)
    data = []
    data_head = [["index", "description", "diagnosis"]]
    for i, batch_data in enumerate(data_loader):
        ids, input_ids, masks = (t.type(torch.LongTensor).to(device) for t in batch_data)
        # Generate Summary
        summary_ids = model.generate(input_ids.to(device), num_beams=args.num_beams, max_length=96)
        # todo: 此处删去了 attention_mask=masks.to(device) 参数，在 input_ids.to(device, ...) 中
        ids = ids.tolist()
        summary_ids = summary_ids.tolist()
        # print(ids)
        for id, summary in zip(ids, summary_ids):
            # turn list of len 1 into number
            id_num = int(id[0])
            # print summary
            end = 2
            while end < len(summary):
                if summary[end] == 1 or summary[end] == 2:
                    break
                end += 1
            summary = summary[2:end]
            # data.append([id, ' '.join(str(x) for x in summary)])
            data_desc_str = ' '.join(str(x) for x in data_desc[id_num][1])
            diagnosis = ' '.join(str(x) for x in summary)
            data.append([id_num, data_desc_str, diagnosis])
    data = sorted(data, key=lambda x: x[0])
    data = data_head + data
    data = pd.DataFrame(data)
    print(data)
    data.to_csv('./report.csv', index=False, header=False)

def main(args):
    device = torch.device(args.device)
    # 读取测试数据
    data = process_data('./', 'data/test.csv', mode='test')
    # data_desc = [row[1] for row in data]
    print('Successfully read data!')
    # 加载测试集 dataloader
    test_loader = pred_dataloader(args, data)
    print('Data loaded successfully!')
    # 加载模型
    model = Seq2SeqModel(args.model_name, args.vocab_size).to(device)
    print('Model initialization successful!')

    if os.path.exists(args.filename):
        print(args.filename)
        # 加载模型权重
        model.load_state_dict(torch.load(args.filename))
        print('Loaded pre-trained model\n')
    else:
        print('Not loaded pre-trained model\n')
    # 解码
    print('Start decoding!')
    decode(model.bart_model, test_loader, args, device, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    preprocess_data_opts(parser)
    train_opts(parser)
    model_opts(parser)

    parser.add_argument('--vocab_size', default=1300, help='')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--filename', default='./model/model/facebook/bart-base-4.pth',
                        help='')

    parser.add_argument('--num_beams', default=10, help='')
    opt = parser.parse_args()
    opt.batch_size = 64
    main(opt)
