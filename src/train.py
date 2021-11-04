import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
import logging
from utils import utils
import math
import json
import torch.distributed as dist
import metrics


def save(model, path, step):
    path += "_epoch{}.pkl".format("{}".format(step))
    logging.info("Save model")
    torch.save(model, path)


def Base_predict(test_iter, model, tokenizer, args):
    raise("predict code")


def Base_valid(valid_iter, model, args):
    model.eval()
    LOSS_fn = nn.CrossEntropyLoss(reduction="none")
    for item in valid_iter:
        input_ids = item["input_ids"].to(args.device)
        input_mask = item["input_mask"].to(args.device)
        if args.train_type == "ner":
            ner_labels = item["ner_labels"].to(args.device)
            ner_labels_mask = (ner_labels != -100)
            lm_logits = model(input_ids, input_mask)
            batch_size, seq_len, class_num = lm_logits.shape
            loss = LOSS_fn(lm_logits.view(-1, class_num), ner_labels.view(-1)).view(batch_size, seq_len)
            loss = torch.mul(loss, ner_labels_mask)
            loss = torch.mean(loss)
            lm_logits = torch.max(F.softmax(lm_logits, dim=-1), dim=-1)[1].reshape(batch_size)
            ner_labels = item["ner_labels"].reshape(batch_size)
            metrics.ner_metrics(lm_logits, ner_labels)
            
        elif args.train_type == "sc":
            sc_labels = item["sc_labels"].to(args.device)
            lm_logits = model(input_ids, input_mask)
            batch_size, class_num = lm_logits.shape
            loss = LOSS_fn(lm_logits.view(-1, class_num), sc_labels.view(-1)).view(batch_size)
            loss = torch.mean(loss)
            lm_logits = torch.max(F.softmax(lm_logits, dim=-1), dim=-1)[1].reshape(batch_size)
            sc_labels = item["ner_labels"].reshape(batch_size)
            metrics.sc_metrics(lm_logits, sc_labels)


def Base_train(train_iter, valid_iter, model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    high_lr = ["classification"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and not any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
            # "lr": args.learning_rate * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            # "lr": args.learning_rate * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) // args.opt_step, num_training_steps=len(train_iter) * args.epoch // args.opt_step)
    mean_loss = 0
    LOSS_fn = nn.CrossEntropyLoss(reduction="none")
    for step in range(args.epoch):
        if args.local_rank != -1:
            train_iter.sampler.set_epoch(step)
        model.train()
        logging.info("Starting Training epoch:{}".format(step+1))
        for idx, item in enumerate(train_iter):
            input_ids = item["input_ids"].to(args.device)
            input_mask = item["input_mask"].to(args.device)
            if args.train_type == "ner":
                ner_labels = item["ner_labels"].to(args.device)
                ner_labels_mask = (ner_labels != -100)
                lm_logits = model(input_ids, input_mask)
                batch_size, seq_len, class_num = lm_logits.shape
                loss = LOSS_fn(lm_logits.view(-1, class_num), ner_labels.view(-1)).view(batch_size, seq_len)
                loss = torch.mul(loss, ner_labels_mask)
                loss = torch.mean(loss)
            elif args.train_type == "sc":
                sc_labels = item["sc_labels"].to(args.device)
                lm_logits = model(input_ids, input_mask)
                batch_size, class_num = lm_logits.shape
                loss = LOSS_fn(lm_logits.view(-1, class_num), sc_labels.view(-1)).view(batch_size)
                loss = torch.mean(loss)
            loss.backward()
            mean_loss += loss.cpu().item()
            if idx % args.opt_step == args.opt_step - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        args.step = step + 1
        mean_loss /= len(train_iter)
        logging.info("Train loss:{:.4f}".format(mean_loss))
        mean_loss = 0
        # if dist.get_rank() == 0:
        with torch.no_grad():
            Base_valid(valid_iter, model, args)
