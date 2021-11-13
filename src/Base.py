import torch
from config import *
import utils
import logging
# from torch._C import dtype
# from transformers.utils.dummy_pt_objects import BartForCausalLM, BartModel
from transformers import AutoTokenizer, BertForSequenceClassification
from model import NER_NET, SC_NET, BERT_CRF
from train import *
import random
import datetime
import csv
import copy
logging.getLogger().setLevel(logging.INFO)
LABEL_LIST = ["O", "B-BANK", "I-BANK", "B-PRODUCT", "I-PRODUCT", "B-COMMENTS_N", "I-COMMENTS_N", "B-COMMENTS_ADJ", "I-COMMENTS_ADJ"]


class Example(object):
    def __init__(self, id, text, labels, sc):
        self.id = id
        self.text = text
        self.labels = labels
        self.sc = sc
        self.len = len(self.text)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer, is_train=False, is_predict=False):
        super(BaseDataset, self).__init__()
        self.input_ids = []
        self.input_mask = []
        self.ner_labels = []
        self.sc_labels = []
        self.is_train = is_train
        self.labels_list = LABEL_LIST
        self.labels_mp = {label: idx for idx, label in enumerate(self.labels_list)}
        if is_predict:
            self.build_predict(Examples, tokenizer)
        else:
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
            # if labels_ids[-1] % 2 == 0 and labels_ids[-1] != 0:
            #     labels_ids[-1] /= 2
            # elif labels_ids[-1] % 2 == 1 and labels_ids[-1] != 0:
            #     labels_ids[-1] = labels_ids[-1] / 2 + 1
        # utils.debug("labels_ids", labels_ids)
        return labels_ids

    def build(self, Examples, tokenizer):
        for item in Examples:
            input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(item.text) + ["[SEP]"])
            input_mask = [1] * len(input_ids)
            ner_labels = self.deocder(item.labels)
            TIME = 1
            if self.is_train:
                if int(item.sc) == 0:
                    TIME = 3
                elif int(item.sc) == 1:
                    TIME = 8
            TIME = 1
            while TIME > 0:
                TIME -= 1
                self.input_ids.append(copy.deepcopy(input_ids))
                self.input_mask.append(copy.deepcopy(input_mask))
                self.ner_labels.append(copy.deepcopy(ner_labels))
                self.sc_labels.append([int(item.sc)])
            if len(input_ids) != len(ner_labels) + 2:
                logging.warning(f"text: {item.text} \nlen: {len(item.text)}")
                logging.warning(f"ner: {item.labels} \nlen: {len(item.labels)}")

    def build_predict(self, Examples, tokenizer):
        for item in Examples:
            input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(item.text) + ["[SEP]"])
            input_mask = [1] * len(input_ids)
            ner_labels = []
            self.input_ids.append(input_ids)
            self.input_mask.append(input_mask)
            self.ner_labels.append(ner_labels)
            self.sc_labels.append([])
    
    def analisy(self):
        mp = {i: 0 for i in range(3)}
        for item in self.sc_labels:
            mp[item[0]] += 1
        logging.info(mp)


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
                ner_max_pad = max(ner_max_pad, len(p))
        else:
            input_max_pad = self.config["FIX_LENGTH"]
            ner_max_pad = self.config["FIX_LENGTH"]
        for i in range(len(batch)):
            out["input_ids"][i] = out["input_ids"][i] + [self.config["PAD_ID"]] * (input_max_pad - len(out["input_ids"][i]))
            out["input_mask"][i] = out["input_mask"][i] + [0] * (input_max_pad - len(out["input_mask"][i]))
            out["ner_labels"][i] = out["ner_labels"][i] + [-100] * (ner_max_pad - len(out["ner_labels"][i]))
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        out["input_mask"] = torch.tensor(out["input_mask"], dtype=torch.long)
        out["ner_labels"] = torch.tensor(out["ner_labels"], dtype=torch.long)
        out["sc_labels"] = torch.tensor(out["sc_labels"], dtype=torch.long)
        return out 


def prepare_examples(path, is_predict=False):
    data = utils.read_from_csv(path)
    Examples = []
    for item in data:
        # utils.debug("item", item)
        if is_predict:
            Examples.append(Example(id=item[0], text=item[1], labels=[], sc=""))
        else:
            Examples.append(Example(id=item[0], text=item[1], labels=item[2].split(" "), sc=item[3]))
    return Examples


def real(predict):
    pre = -1
    res = []
    for item in predict:
        # logging.info(f"item: {item}")
        res.append(LABEL_LIST[item])
        # if item == 0:
        #     res.append(LABEL_LIST[0])
        #     pre = item
        # else:
        #     if pre == item:
        #         res.append(LABEL_LIST[item * 2])
        #     else:
        #         res.append(LABEL_LIST[item * 2 - 1])
        #     pre = item
    return res


def extend_data(train_data):
    smw_E = utils.build_EquivalentChar()
    smw_R = utils.build_RandomDeleteChar()
    smw_S = utils.build_Similarword()
    smw_H = utils.build_Homophone()
    re_train_data = []
    for item in train_data:
        if int(item.sc) in [0, 1, 2]:
            res = utils.do_nlpcda(smw_E, item.text)
            re_train_data.extend([Example(id=-1, text=str(r), labels=["O"] * len(r), sc=item.sc) for r in res])
        if int(item.sc) in [0, 1, 2]:
            re_train_data.append(item)
        if int(item.sc) in [0, 1]:
            res = utils.do_nlpcda(smw_R, item.text)
            re_train_data.extend([Example(id=-1, text=str(r), labels=["O"] * len(r), sc=item.sc) for r in res])
        if int(item.sc) in [1]:
            res = utils.do_nlpcda(smw_S, item.text)
            re_train_data.extend([Example(id=-1, text=str(r), labels=["O"] * len(r), sc=item.sc) for r in res])
        if int(item.sc) in [0, 1]:
            res = utils.do_nlpcda(smw_H, item.text)
            re_train_data.extend([Example(id=-1, text=str(r), labels=["O"] * len(r), sc=item.sc) for r in res])
    return re_train_data


def main(args):
    logging.info("Config Init")
    torch.cuda.set_device(0)
    args.device = torch.device("cuda", 0)
    # args.device = torch.device("cpu")
    if args.crf:
        LABEL_LIST.extend(["[CLS]", "[SEP]"])
    logging.info("Load Data")
    data = prepare_examples(args.train_path)
    random.shuffle(data)
    train_data = data[:int(len(data) * 0.95)]
    valid_data = data[int(len(data) * 0.95):]
    # if args.train_type == "sc":
    #     train_data = extend_data(train_data)
    # utils.predict_use_senta(valid_data)
    logging.info("Init Model and Tokenizer")
    # args.ner_class = len(LABEL_LIST)
    args.ner_class = len(LABEL_LIST)
    args.sc_class = 3
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.model_load:
        model = torch.load(args.model_load)
    else:
        if args.train_type == "ner":
            if args.model_load:
                model = torch.load(args.model_load)
            else:
                # model = NER_NET(args)
                tag_to_ix = {label: idx for idx, label in enumerate(LABEL_LIST)}
                args.tag_to_ix = tag_to_ix
                model = BERT_CRF(args)
        elif args.train_type == "sc":
            # model = BertForSequenceClassification.from_pretrained(args.pretrain_path)
            # model.config.num_labels = 3
            if args.model_load:
                model = torch.load(args.model_load)
            else:
                model = SC_NET(args)
    word_token = ["“", "”", "-"]
    tokenizer.add_tokens(word_token)
    tokenizer.eos_token = "[SEP]"
    tokenizer.bos_token = "[CLS]"
    if args.crf:
        model.ner_net.bert.config.eos_token_id = tokenizer.eos_token_id
        model.ner_net.bert.config.bos_token_id = tokenizer.bos_token_id
        model.ner_net.bert.resize_token_embeddings(len(tokenizer))
        model.ner_net.bert.config.device = args.device
        logging.info(f"eos_token_id:{model.ner_net.bert.config.eos_token_id}")
        logging.info(f"bos_token_id:{model.ner_net.bert.config.bos_token_id}")
    else:
        model.bert.config.eos_token_id = tokenizer.eos_token_id
        model.bert.config.bos_token_id = tokenizer.bos_token_id
        model.bert.resize_token_embeddings(len(tokenizer))
        model.bert.config.device = args.device
        logging.info(f"eos_token_id:{model.bert.config.eos_token_id}")
        logging.info(f"bos_token_id:{model.bert.config.bos_token_id}")
    model = model.to(args.device)
    args.pad_id = tokenizer.pad_token_id
    logging.info("Prepare Dataset")
    train_dataset = BaseDataset(train_data, tokenizer, is_train=True)
    valid_dataset = BaseDataset(valid_data, tokenizer)
    train_dataset.analisy()
    valid_dataset.analisy()

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collection(args))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    logging.info("Start Training")
    Base_train(train_iter, valid_iter, model, args)


def predict(args):
    logging.info("Config Init")
    torch.cuda.set_device(0)
    # dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", 0)
    logging.info("Load Data")
    test_data = prepare_examples(args.test_path, is_predict=True)
    logging.info("Init Model and Tokenizer")
    tokenizer_sc = AutoTokenizer.from_pretrained(args.tokenizer_sc_path)
    tokenizer_ner = AutoTokenizer.from_pretrained(args.tokenizer_ner_path)
    ner_model = torch.load(args.ner_model_load).to(args.device)
    sc_model = torch.load(args.sc_model_load).to(args.device)
    word_token = ["“", "”", "-"]
    tokenizer_sc.add_tokens(word_token)
    tokenizer_sc.pad_token = "[PAD]"
    tokenizer_sc.eos_token = "[SEP]"
    tokenizer_sc.bos_token = "[CLS]"
    args.pad_id = tokenizer_sc.pad_token_id
    tokenizer_ner.add_tokens(word_token)
    tokenizer_ner.pad_token = "[PAD]"
    tokenizer_ner.eos_token = "[SEP]"
    tokenizer_ner.bos_token = "[CLS]"
    sc_test_dataset = BaseDataset(test_data, tokenizer_sc, args, is_predict=True)
    ner_test_dataset = BaseDataset(test_data, tokenizer_ner, args, is_predict=True)
    sc_test_iter = torch.utils.data.DataLoader(sc_test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    args.pad_id = tokenizer_ner.pad_token_id
    if args.crf:
        args.batch_size = 1
    ner_test_iter = torch.utils.data.DataLoader(ner_test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    logging.info("Start predict")
    with torch.no_grad():
        args.predict_type = "ner"
        ner_res = Base_predict(ner_test_iter, ner_model, args)
        args.predict_type = "sc"
        if args.ensemble:
            models = [torch.load(args.model1).to(args.device), torch.load(args.model2).to(args.device), \
                torch.load(args.model3).to(args.device)]
            sc_res = Base_predict_ensemble(sc_test_iter, models, args)
        else:
            sc_res = Base_predict(sc_test_iter, sc_model, args)
        logging.info(f"ner_se len: {len(ner_res)}")
        logging.info(f"sc_res len: {len(sc_res)}")
        assert len(ner_res) == len(sc_res)
    output = []
    for i in range(len(test_data)):
        line = test_data[i]
        line.labels = real(ner_res[i][:line.len])
        line.sc = int(sc_res[i])
        output.append(line)
    with open(args.output_path, "w", encoding="utf-8") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["id", "BIO_anno", "class"])
        for item in output:
            csv_write.writerow([item.id, " ".join(item.labels), item.sc])
    logging.info("END")


if __name__ == "__main__":
    args = Base_config()
    utils.set_seed(args.seed)    
    if args.train:
        args.model_save = '/'.join([args.model_save, utils.d2s(datetime.datetime.now(), time=True)])
        logging.info(f"model_save: {args.model_save}")
        main(args)
    if args.predict:
        predict(args)
