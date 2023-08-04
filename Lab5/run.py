import argparse

from transformers import BertTokenizer

from models.model import mymodel
from models.resnet import resnet152, myResnet
from processors.util import *

labelMap = {0: "negative", 1: "neutral", 2: "positive"}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data_txt', type=str, help="directory that contains processed data.")
    parser.add_argument("--train_file", default='train.tsv', type=str, help="file that contains train data.")
    parser.add_argument("--valid_file", default='valid.tsv', type=str, help="file that contains valid data.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="Bert pre-trained model")
    parser.add_argument("--num_labels", default=3, type=int, help="number of label")
    parser.add_argument("--output_dir", default='./output', type=str, help="directory that contains results.")
    parser.add_argument("--text_only", default=0, type=int, help="进行消融实验，只输入文字")
    parser.add_argument("--image_only", default=0, type=int, help="进行消融实验，只输入图片")
    parser.add_argument("--max_seq_length", default=64, type=int, help="the max length of the input sequence")
    parser.add_argument("--do_train", default=1, type=int, help="Whether to run training.")
    parser.add_argument("--do_eval", default=1, type=int, help="Whether to run eval on the valid set.")
    parser.add_argument("--do_test", default=1, type=int, help="Whether to predict on the test set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,help="warmup_proportion value.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay value")

    parser.add_argument("--no_cuda", action='store_true', help="是否使用cuda")
    parser.add_argument('--seed', type=int, default=100, help="random seed")
    parser.add_argument('--fine_tune_cnn', action='store_true', help='是否fine tune cnn')
    parser.add_argument('--resnet_root', default='./pre_trained_model', help='存放resnet预训练模型的路径')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
    parser.add_argument('--path_image', default="./data/", help='图像的存放路径')

    args = parser.parse_args()
    setseed(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # 如果output_dir不存在，新建文件夹
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    label_list = []
    for i in range(args.num_labels):
        label_list.append(str(i))

    train_examples, num_train_steps = None, None
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # 进行训练
    model = mymodel.from_pretrained(args.bert_model, num_labels=args.num_labels)

    net = resnet152()
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
    encoder = myResnet(net, args.fine_tune_cnn, args.device)
    model.to(args.device)
    encoder.to(args.device)
    if args.do_train:
        train_examples = create_examples(read_tsv(os.path.join(args.data_dir, args.train_file)), "train", args.text_only,
                                         args.image_only)
        # train_examples=train_examples[:100]
        print(len(train_examples))
        num_train_steps = int(len(train_examples) / args.train_batch_size * args.num_train_epochs)
        optimizer, scheduler = get_setting(args, num_train_steps, model)
    global_step = 0
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")
    if args.do_train:
        train(args, train_examples, num_train_steps, label_list, optimizer, scheduler, model, encoder, global_step, tokenizer)

    # 进行预测
    if args.do_test:
        model_state_dict = torch.load(output_model_file)
        model = mymodel.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=args.num_labels)
        model.to(args.device)
        encoder_state_dict = torch.load(output_encoder_file)
        encoder.load_state_dict(encoder_state_dict)
        encoder.to(args.device)
        test(args, label_list, tokenizer, model, encoder, labelMap)


if __name__ == "__main__":
    main()