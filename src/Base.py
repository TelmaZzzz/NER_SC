import torch
from config import *
import utils
import logging
# from torch._C import dtype
# from transformers.utils.dummy_pt_objects import BartForCausalLM, BartModel
from transformers import AutoTokenizer
from model import NER_NET, SC_NET
from train import *
import json
import random
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
logging.getLogger().setLevel(logging.INFO)
LABEL_LIST = ["O", "B-BANK", "I-BANK", "B-PRODUCT", "I-PRODUCT", "B-COMMENTS_N", "I-COMMENTS_N", "B-COMMENTS_ADJ", "I-COMMENTS_ADJ"]


class Example(object):
    def __init__(self, id, text, labels, sc):
        self.id = id
        self.text = text
        self.labels = labels
        self.sc = sc


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer):
        super(BaseDataset, self).__init__()
        self.input_ids = []
        self.input_mask = []
        self.ner_labels = []
        self.sc_labels = []
        self.labels_list = ["O", "B-BANK", "I-BANK", "B-PRODUCT", "I-PRODUCT", "B-COMMENTS_N", "I-COMMENTS_N", "B-COMMENTS_ADJ", "I-COMMENTS_ADJ"]
        self.labels_mp = {label: idx for idx, label in enumerate(self.labels_list)}
        self.build(Examples, tokenizer)
 
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "input_mask": self.input_mask[idx],
            "ner_labels": self.ner_labels[idx],
            "sc_labels": self.sc_labels[idx]
        }
    
    def __len__(self):
        return len(self.input_ids)
    
    def deocder(self, labels):
        labels_ids = []
        for label in labels:
            labels_ids.append(self.labels_mp[label])
        return labels_ids

    def build(self, Examples, tokenizer):
        for item in Examples:
            input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS]" + item.text + "[SEP]"))
            input_mask = [1] * len(input_ids)
            ner_labels = self.decoder(item.ner_labels)
            self.input_ids.append(input_ids)
            self.input_mask.append(input_mask)
            self.ner_labels.append(ner_labels)
            self.sc_labels.append([item.sc])


class Collection(object):
    def __init__(self, args):
        self.config = {}
        self.config["BUCKET"] = True
        self.config["FIX_LENGTH"] = args.fix_length
        self.config["PAD_ID"] = args.pad_id

    def __call__(self, batch):
        out = {
            "input_ids": [],
            "input_mask": [],
            "ner_labels": [],
            "sc_labels": [],
        }
        for mini_batch in batch:
            for k, v in mini_batch.items():
                out[k].append(v)
        input_max_pad = 0
        ner_max_pad = 0
        if self.config["BUCKET"]:
            for p in out["input_ids"]:
                input_max_pad = max(input_max_pad, len(p))
            for p in out["ner_labels"]:
                output_max_pad = max(output_max_pad, len(p))
        else:
            input_max_pad = self.config["FIX_LENGTH"]
            output_max_pad = self.config["FIX_LENGTH"]
        for i in range(len(batch)):
            out["input_ids"][i] = out["input_ids"][i] + [self.config["PAD_ID"]] * (input_max_pad - len(out["input_ids"][i]))
            out["input_mask"][i] = out["input_mask"][i] + [0] * (input_max_pad - len(out["input_mask"][i]))
            out["ner_labels"][i] = out["ner_labels"][i] + [-100] * (ner_max_pad - len(out["ner_labels"][i]))
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        out["input_mask"] = torch.tensor(out["input_mask"], dtype=torch.long)
        out["ner_labels"] = torch.tensor(out["ner_labels"], dtype=torch.long)
        out["sc_labels"] = torch.tensor(out["sc_labels"], dtype=torch.long)
        return out 


def prepare_examples(path):
    data = utils.read_from_csv(path)
    Examples = []
    for item in data:
        Examples.append(Example(id=item[0], text=item[1], labels=item[2].split(" "), sc=item[3]))
    return Examples


def main(args):
    logging.info("Config Init")
    torch.cuda.set_device(0)
    args.device = torch.device("cuda", 0)
    logging.info("Load Data")
    data = prepare_examples(args.train_path)
    random.shuffle(data)
    train_data = data[:int(len(data) * 0.95)]
    valid_data = data[int(len(data) * 0.95):]
    logging.info("Init Model and Tokenizer")
    args.ner_class = len(LABEL_LIST)
    args.sc_class = 3
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.model_load:
        model = torch.load(args.model_load)
    else:
        if args.train_type == "ner":
            model = NER_NET(args)
        elif args.train_type == "sc":
            model = SC_NET(args)
    word_token = ["“", "”", "-"]
    tokenizer.add_tokens(word_token)
    tokenizer.eos_token = "[SEP]"
    tokenizer.bos_token = "[CLS]"
    model.bert.config.eos_token_id = tokenizer.eos_token_id
    model.bert.config.bos_token_id = tokenizer.bos_token_id
    model.bert.resize_token_embeddings(len(tokenizer))
    model.bert.config.device = args.device
    logging.info(f"eos_token_id:{model.config.eos_token_id}")
    logging.info(f"bos_token_id:{model.config.bos_token_id}")
    logging.info(f"gpu num:{args.n_gpu}")
    # DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    logging.info(f"local rank:{args.local_rank}")
    model = model.to(args.device)
    args.pad_id = tokenizer.pad_token_id
    logging.info("Prepare Dataset")
    train_dataset = BaseDataset(train_data, tokenizer)
    valid_dataset = BaseDataset(valid_data, tokenizer)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collection(args))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    # test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    logging.info("Start Training")
    Base_train(train_iter, valid_iter, model, args)


def predict(args):
    logging.info("Config Init")
    torch.cuda.set_device(0)
    # dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", 0)
    logging.info("Load Data")
    test_data = prepare_examples(args.test_path)
    logging.info("Init Model and Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = torch.load(args.model_load).to(args.device)
    word_token = ["“", "”", "-"]
    tokenizer.add_tokens(word_token)
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.bos_token = "[CLS]"
    args.pad_id = tokenizer.pad_token_id
    test_dataset = BaseDataset(test_data, tokenizer)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    logging.info("Start predict")
    parameter_list = utils.get_parameter()
    continue_list = []
    args.output += f"_batch{args.batch_size}"
    with torch.no_grad():
        for idx, parameter in enumerate(parameter_list):
            if idx in continue_list:
                continue
            args.parameter = parameter
            args.step = idx
            Base_predict(test_iter, model, args)
    logging.info("END")


if __name__ == "__main__":
    args = Base_config()
    utils.set_seed(959794)
    if args.train:
        args.model_save = '/'.join([args.model_save, utils.d2s(datetime.datetime.now(), time=True)])
        main(args)
    if args.predict:
        args.output = '/'.join([args.output, utils.d2s(datetime.datetime.now(), time=True)])
        predict(args)