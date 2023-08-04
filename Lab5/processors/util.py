import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_recall_fscore_support
import logging
import csv
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import random
import torch
import warnings


warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def calculate_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro = precision_recall_fscore_support(true, preds, average='weighted')
    return f_macro

def setseed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_setting(args, t_total, model):
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.1),
                                                num_training_steps=t_total)
    return optimizer, scheduler




def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

class InputExample(object):
    def __init__(self, guid, text=None, img_id=None, label=None):
        self.guid = guid
        self.text = text
        self.img_id = img_id
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids,img_feat, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.img_feat = img_feat
        self.label_id = label_id


def read_tsv(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def create_examples(lines, set_type, text_only, image_only):
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text = line[3].lower()
        img_id = line[2]
        label = line[1]
        if text_only == 1:
            img_id = None
        if image_only == 1:
            text = None
        examples.append(
            InputExample(guid=guid, text=text, img_id=img_id, label=label))
    return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, crop_size,path_img):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    # 默认是224 * 224
    transform = transforms.Compose([
        transforms.RandomCrop(crop_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    input_ids, input_mask, added_input_mask, segment_ids= None, None, None, None
    for (ex_index, example) in enumerate(examples):
        if example.text:
            tokens_a = tokenizer.tokenize(example.text)
            while True:
                t_len = len(tokens_a) + len(tokens_a)
                if t_len <= max_seq_length - 3:
                    break
                tokens_a.pop()

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            tokens += tokens_a + ["[SEP]"]
            segment_ids += [1] * (len(tokens_a) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            added_input_mask = [1] * (len(input_ids) + 49)  # for encoding regional image representations
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            added_input_mask += padding
            segment_ids += padding

        if example.label == "null":
            label_id = None
        else:
            label_id = label_map[example.label]
        image = None
        if example.img_id:
            image_name = example.img_id + ".jpg"
            image_path = os.path.join(path_img, image_name)
            if not os.path.exists(image_path):
                print("image_path do not exist!")
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                          segment_ids=segment_ids,img_feat=image, label_id=label_id))
    return features

def train(args,train_examples, num_train_steps, label_list,optimizer,scheduler,model,encoder,global_step,tokenizer):
    if args.do_train:
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, args.crop_size,
            args.path_image)
        if args.image_only == 0 and args.text_only == 0:
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            all_img_feats = torch.stack([f.img_feat for f in train_features])
            train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,all_img_feats, all_label_ids)
        elif args.text_only == 1:
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,all_label_ids)
        elif args.image_only == 1:
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            all_img_feats = torch.stack([f.img_feat for f in train_features])
            train_data = TensorDataset(all_img_feats, all_label_ids)

        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        eval_examples = create_examples(read_tsv(os.path.join(args.data_dir, args.valid_file)), "valid", args.text_only,
                                        args.image_only)
        # eval_examples=eval_examples[:10]
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.crop_size,
            args.path_image)

        if args.text_only == 0 and args.image_only == 0:
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            all_img_feats = torch.stack([f.img_feat for f in eval_features])
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,all_img_feats, all_label_ids)
        elif args.text_only == 1:
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,all_label_ids)
        elif args.image_only == 1:
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            all_img_feats = torch.stack([f.img_feat for f in eval_features])
            eval_data = TensorDataset(all_img_feats, all_label_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        max_acc = 0.0
        logger.info("Start Training")
        logger.info("Batch size = %d", args.train_batch_size)
        for train_idx in range(args.num_train_epochs):
            model.train()
            encoder.train()
            encoder.zero_grad()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            step = 0
            for batch in bar:
                batch = tuple(t.to(args.device) for t in batch)
                if args.text_only == 1:
                    input_ids, input_mask, added_input_mask, segment_ids, label_ids = batch
                elif args.text_only == 0 and args.image_only == 0:
                    input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids = batch
                elif args.image_only == 1:
                    img_feats, label_ids = batch
                with torch.no_grad():
                    if args.text_only == 1:
                        pass
                    else:
                        # 先用resnet对图像特征进行提取
                        imgs_f, img_mean, img_att = encoder(img_feats)
                if train_idx == 0 and step == 0:
                    if args.text_only == 1:
                        img_att = None
                    if args.image_only==1:
                        input_ids,segment_ids,input_mask,added_input_mask=None,None,None,None
                    loss = model(input_ids, img_att, segment_ids, input_mask,added_input_mask, label_ids)
                else:
                    if args.text_only == 1:
                        img_att = None
                    if args.image_only == 1:
                        input_ids, segment_ids, input_mask, added_input_mask = None, None, None, None
                    loss = model(input_ids, img_att, segment_ids, input_mask, added_input_mask, label_ids)

                loss.backward()
                scheduler.step()

                tr_loss += loss.item()
                if input_ids != None:
                    nb_tr_examples += input_ids.size(0)
                else:
                    nb_tr_examples += img_att.size(0)
                nb_tr_steps += 1

                if (step + 1) % 1 == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                step = step + 1
                avg_loss = tr_loss / step
                bar.set_description("[epoch {}] training loss {:.4f}".format(train_idx, avg_loss))

            logger.info("Start Evaluation")
            model.eval()
            encoder.eval()
            eval_loss, eval_accuracy = 0, 0
            eval_steps, eval_numbers = 0, 0

            true_label_list, pred_label_list = [], []

            bar = tqdm(eval_dataloader, total=len(eval_dataloader))
            for batch in bar:
                batch = tuple(t.to(args.device) for t in batch)
                if args.text_only == 1:
                    input_ids, input_mask, added_input_mask, segment_ids,label_ids = batch
                elif args.text_only == 0 and args.image_only == 0:
                    input_ids, input_mask, added_input_mask, segment_ids,img_feats, label_ids = batch
                elif args.image_only == 1:
                    img_feats, label_ids = batch
                with torch.no_grad():
                    if args.text_only == 1:
                        img_att = None
                    else:
                        if args.image_only == 1:
                            input_ids, segment_ids, input_mask, added_input_mask = None, None, None, None
                        imgs_f, img_mean, img_att = encoder(img_feats)
                    tmp_eval_loss = model(input_ids, img_att, segment_ids,input_mask, added_input_mask, label_ids)
                    logits = model(input_ids, img_att, segment_ids, input_mask, added_input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                outputs = np.argmax(logits, axis=1)
                tmp_eval_accuracy = np.sum(outputs == label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
                if input_ids != None:
                    eval_numbers += input_ids.size(0)
                else:
                    eval_numbers += img_att.size(0)
                eval_steps += 1
                avg_loss = eval_loss / eval_steps
                bar.set_description("[epoch {}] valid loss {:.4f}".format(train_idx, avg_loss))

            eval_accuracy = eval_accuracy / eval_numbers
            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            F_score = calculate_f1(true_label, pred_outputs)
            print('eval_accuracy: {:.4f} , f_score: {:.4f}'.format(eval_accuracy, F_score))

            if eval_accuracy >= max_acc:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder  # Only save the model it-self
                torch.save(model_to_save.state_dict(), output_model_file)
                torch.save(encoder_to_save.state_dict(), output_encoder_file)
                max_acc = eval_accuracy

def test(args,label_list,tokenizer,model,encoder,labelMap):
    eval_examples = create_examples(read_tsv(os.path.join(args.data_dir, "test.tsv")), "test",0,0)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, args.crop_size,args.path_image)
    logger.info("Start Prediction")
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_img_feats = torch.stack([f.img_feat for f in eval_features])
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,all_img_feats)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    encoder.eval()
    pred_label_list = []

    for input_ids, input_mask, added_input_mask, segment_ids, img_feats in tqdm(eval_dataloader, desc="Testing"):
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        added_input_mask = added_input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        img_feats = img_feats.to(args.device)

        with torch.no_grad():
            imgs_f, img_mean, img_att = encoder(img_feats)
            logits = model(input_ids, img_att, segment_ids, input_mask, added_input_mask)

        logits = logits.detach().cpu().numpy()
        pred_label_list.append(logits)

    pred_outputs = np.concatenate(pred_label_list)
    pred_label = np.argmax(pred_outputs, axis=-1)

    test_file = os.path.join("./data_txt/", "test_without_label.txt")
    f_test = open(test_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines_test = f_test.readlines()
    fp = open(os.path.join(args.output_dir, "test_without_label.txt"), 'w')
    for i in range(0, len(lines_test)):
        if i == 0:
            fp.write(lines_test[i])
            continue
        guid, tag = lines_test[i].split(",")
        tag = labelMap[pred_label[i - 1]]
        fp.write(guid + "," + tag + '\n')
    fp.close()
    f_test.close()