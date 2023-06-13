import torch
from data.my_dataset import PredDataset, BartDataset

def pred_dataloader(args, meta_data, shuffle=True):
    data_dataset = PredDataset(meta_data, encode_max_length=args.encode_max_length, decode_max_length=args.decode_max_length, vocab_size=args.vocab_size)
    data_loader = torch.utils.data.DataLoader(data_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=False,
                                              num_workers=args.NUM_WORKERS)
    return data_loader

def build_pre_bart_dataloader(args, meta_data, shuffle=True):
    data_dataset = BartDataset(meta_data, max_length=args.encode_max_length, vocab_size=args.vocab_size)
    data_loader = torch.utils.data.DataLoader(data_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              num_workers=args.NUM_WORKERS)
    return data_loader
