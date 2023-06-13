import argparse
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from data.process_data import process_data
from data.dataloader import build_pre_bart_dataloader
from model.small_bart import Seq2SeqModel
from utils.build_optimizer import build_optimizer
from utils.build_lr_scheduler import build_scheduler
from tqdm import tqdm
from utils.save_model import save_model
from opts import *
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")
    device = torch.device(args.device)
    set_seed(seed=args.seed)
    tb_writer = SummaryWriter()
    model_dir = os.path.join(args.result_dir, 'model')
    if os.path.exists(model_dir) is False:
        os.makedirs(model_dir)
    # Data processing
    data = process_data('./', 'data/train.csv', mode='train')
    test_data = process_data('./', 'data/test.csv', mode='test')

    # split train.csv to train_data and valid_data
    n_split = int(len(data) * 0.8)
    train_data = data[:n_split]
    valid_data = data[n_split:]
    print('Train data length: ', len(train_data))
    print('Valid data length: ', len(valid_data))
    print('Test data length: ', len(test_data))
    # Load data
    train_loader = build_pre_bart_dataloader(args, train_data, shuffle=True)
    valid_loader = build_pre_bart_dataloader(args, valid_data, shuffle=True)
    # Load model
    model = Seq2SeqModel(args.model_name, args.vocab_size).to(device)
    if args.filename is not None and os.path.exists(args.filename):
        # 加载模型权重
        model.load_state_dict(torch.load(args.filename))
        print('Loaded pre-trained model:', args.filename)
    else:
        print('Not loaded pre-trained model.\n')

    pg = [p for p in model.parameters() if p.requires_grad]
    print('Model loaded successfully.')
    # Define optimizer and scheduler
    optimizer = build_optimizer(args.optimizer_name, pg, args.lr, args.weight_decay)
    print('Optimizer configured successfully.')
    scheduler = build_scheduler(args.lr_decay_func, optimizer, args.num_warmup_steps, args.epochs, args.lrf)

    scaler = torch.cuda.amp.GradScaler()
    # 混合精度训练
    for epoch in range(args.epochs):
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('lr:', cur_lr)
        print('weight decay:', optimizer.state_dict()['param_groups'][0]['weight_decay'])
        train_loss = train_one_epoch(train_loader, model, optimizer, epoch, device=device, is_adversial=False, scaler=scaler)
        valid_loss = valid(valid_loader, model, device, epoch, args.num_beams, args.file_valid)

        tb_writer.add_scalar('train loss', train_loss, epoch)
        tb_writer.add_scalar('valid loss', valid_loss, epoch)
        tb_writer.add_scalar('lr', cur_lr, epoch)
        if (epoch + 1) % 5 == 0:
            save_model(model, model_dir, args.model_name, epoch)
        scheduler.step()

    tb_writer.close()

def train_one_epoch(train_loader, model, optimizer, epoch, device, is_adversial, scaler):
    model.train()
    num_steps = len(train_loader)
    mean_loss = torch.zeros(1).to(device)
    mean_lm_loss = torch.zeros(1).to(device)
    mean_mlm_loss = torch.zeros(1).to(device)
    # 在进程 0 中打印训练进度
    data_loader = tqdm(train_loader, total=num_steps)
    for idx, batch in enumerate(data_loader):
        enc_inputs, enc_attention_mask, dec_inputs, dec_labels, dec_masks = \
            (t.type(torch.LongTensor).to(device) for t in batch)
        with torch.cuda.amp.autocast():
            loss, lm_loss, mlm_loss = model(enc_inputs, mlm_labels=dec_labels,masks=enc_attention_mask,
                                            decoded_inputs=dec_inputs, lm_labels=dec_labels, dec_masks=dec_masks)
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Clip the norm of the gradients to 1
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        mean_loss = (mean_loss * idx + loss.item()) / (idx + 1)  # update mean losses
        mean_lm_loss = (mean_lm_loss * idx + lm_loss.item()) / (idx + 1)  # update mean losses
        mean_mlm_loss = (mean_mlm_loss * idx + mlm_loss.item()) / (idx + 1)  # update mean losses

        data_loader.desc = "[train epoch {}] loss: {:.3f} lm loss: {:.3f} mlm loss: {:.3f}"\
            .format(epoch, mean_loss.item(), mean_lm_loss.item(), mean_mlm_loss.item())
    return mean_loss


def valid(valid_loader, model, device, epoch, num_beams, file_valid):
    model.eval()
    print('Loaded pre-trained model:', file_valid)
    model.load_state_dict(torch.load(file_valid))
    num_steps = len(valid_loader)
    valid_loader = tqdm(valid_loader, total=num_steps)
    mean_loss = torch.zeros(1).to(device)
    for i, batch in enumerate(valid_loader):
        # enc_inputs, dec_inputs, labels, attention_mask = (t.type(torch.LongTensor).to(device) for t in batch)
        enc_inputs, enc_attention_mask, dec_inputs, dec_labels, dec_masks = \
            (t.type(torch.LongTensor).to(device) for t in batch)
        # 删除 dec_labels 末尾的所有 0
        # 找到最后的不为零的元素索引
        # 去掉末尾所有的零
        dec_labels_short = []
        for dec_label in dec_labels:
            dec_label = dec_label[:-1]
            # 在函数内部，我们使用 nonzero() 函数来查找张量中所有非零元素的索引。as_tuple=True 参数使函数返回一个元组，
            # 其中包含两个张量：一个包含非零元素的行索引，另一个包含非零元素的列索引。由于我们只需要行索引，
            # 因此我们使用 [0] 来提取它们。
            dec_label_nonzero = dec_label[dec_label.nonzero(as_tuple=True)[0]].tolist()
            # print('printing dec_label_nonzero:')
            # print(dec_label_nonzero)
            dec_label_nonzero = [str(word) for word in dec_label_nonzero][:-1]
            dec_labels_short.append(dec_label_nonzero)
        # print('printing dec_labels_short:')
        # print(dec_labels_short)
        model_output = model.bart_model.generate(enc_inputs.to(device), num_beams=num_beams, max_length=96)
        model_output_short = []
        for item in model_output:
            item = item[1:-1]
            item_nonzero = item[item.nonzero(as_tuple=True)[0]].tolist()
            item_nonzero = [str(word) for word in item_nonzero][:-1]
            model_output_short.append(item_nonzero)
        model_output_short = [[str(word) for word in sentence] for sentence in model_output_short]

        bleu_score = corpus_bleu(dec_labels_short, model_output_short)
        print('bleu score is:', bleu_score)
        # bleu_score = corpus_bleu(dec_labels_short, model_output)
        with torch.cuda.amp.autocast():
            loss, lm_loss, mlm_loss = model(enc_inputs, masks=enc_attention_mask, decoded_inputs=dec_inputs,
                                                lm_labels=dec_labels, dec_masks=dec_masks)
        mean_loss = (mean_loss * i + loss.item()) / (i + 1)  # update mean losses
        valid_loader.desc = "[valid epoch {}] loss: {:.3f} bleu_score: {:.4f}"\
            .format(epoch, mean_loss.item(), bleu_score)
    return mean_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    preprocess_data_opts(parser)
    model_opts(parser)
    batch_size_opts(parser)
    train_opts(parser)
    optimizer_opts(parser)
    scheduler_opts(parser)

    parser.add_argument('--seed', default=7, help='')
    parser.add_argument('--vocab_size', default=1300, help='')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--is_need_frozen', default=False, type=int, help='')
    parser.add_argument('--num_beams', default=2, type=int, help='')
    parser.add_argument('--file_valid', default='./model/model/facebook/bart-base-9.pth', help='')
    opt = parser.parse_args()
    opt.filename = None
    main(opt)
