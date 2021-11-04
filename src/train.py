import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
import logging
import utils
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
    score_mean = 0
    total = 0
    for item in valid_iter:
        input_ids = item["input_ids"].to(args.device)
        input_mask = item["input_mask"].to(args.device)
        if args.train_type == "ner":
            ner_labels = item["ner_labels"].to(args.device)
            # ner_labels_mask = (ner_labels != -100)
            lm_logits = model(input_ids, input_mask)
            batch_size, seq_len, class_num = lm_logits.shape
            # loss = LOSS_fn(lm_logits.view(-1, class_num), ner_labels.view(-1)).view(batch_size, seq_len)
            # loss = torch.mul(loss, ner_labels_mask)
            # loss = torch.mean(loss)
            # utils.debug("eval lm_logits 1", lm_logits.shape)
            lm_logits = torch.max(lm_logits, dim=-1)[1].reshape(batch_size * seq_len)
            # utils.debug("eval lm_logits 2", lm_logits)
            ner_labels = ner_labels.reshape(batch_size * seq_len)
            ner_labels_mask = (ner_labels != -100)
            predict = torch.masked_select(lm_logits, ner_labels_mask)
            gold = torch.masked_select(ner_labels, ner_labels_mask)
            utils.debug("predict shape", predict.shape)
            utils.debug("gold shape", gold.shape)
            score = metrics.ner_metrics(predict.cpu(), gold.cpu())
            score_mean += score * batch_size
            total += batch_size
        elif args.train_type == "sc":
            sc_labels = item["sc_labels"].to(args.device)
            lm_logits = model(input_ids, input_mask)
            batch_size, class_num = lm_logits.shape
            # loss = LOSS_fn(lm_logits.view(-1, class_num), sc_labels.view(-1)).view(batch_size)
            # loss = torch.mean(loss)
            lm_logits = torch.max(lm_logits, dim=-1)[1].reshape(batch_size)
            sc_labels = item["ner_labels"].reshape(batch_size)
            score = metrics.sc_metrics(lm_logits.cpu(), sc_labels.cpu())
            score_mean += score * batch_size
            total += batch_size
    score_mean /= total
    logging.info("epoch{} score:{:.4f}".format(args.step, score_mean))
    save(model, args.model_save, args.step)


def Base_train(train_iter, valid_iter, model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    # high_lr = ["classification"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
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
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) // args.opt_step, num_training_steps=len(train_iter) * args.epoch // args.opt_step)
    mean_loss = 0
    LOSS_fn = nn.CrossEntropyLoss(reduction="none")
    for step in range(args.epoch):
        model.train()
        logging.info("Starting Training epoch:{}".format(step+1))
        for idx, item in enumerate(train_iter):
            input_ids = item["input_ids"].to(args.device)
            input_mask = item["input_mask"].to(args.device)
            # utils.debug("input_ids shape", input_ids.shape)
            if args.train_type == "ner":
                ner_labels = item["ner_labels"].to(args.device)
                # utils.debug("ner_labels shape", ner_labels.shape)
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
