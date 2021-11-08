import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
import logging
import utils
import math
import json
import torch.distributed as dist
import metrics
SCOREs = []


def train_model(model, layer):
    for k, v in model.named_parameters():
        if f"layer.{layer}" in k:
            v.requires_grad = True


def force_model(model):
    for k, v in model.named_parameters():
        if "bert" in k:
            v.requires_grad = False


def save(model, path, score):
    path = path + "_score_{:.4f}.pkl".format(score)
    logging.info("Save model")
    torch.save(model, path)


def rm(path, score):
    path = path + "_score_{:.4f}.pkl".format(score)
    if os.path.exists(path):
        os.remove(path)
        logging.info("model remove success!!!")


def Base_predict(test_iter, model, args):
    model.eval()
    predict_list = []
    for item in test_iter:
        input_ids = item["input_ids"].to(args.device)
        input_mask = item["input_mask"].to(args.device)
        if args.predict_type == "ner":
            # ner_labels_mask = (ner_labels != -100)
            lm_logits = model(input_ids, input_mask)
            batch_size, seq_len, class_num = lm_logits.shape
            predict = torch.max(F.softmax(lm_logits, dim=-1), dim=-1)[1].view(batch_size, seq_len)
            predict_list.extend(predict.tolist())
        elif args.predict_type == "sc":
            lm_logits = model(input_ids, input_mask)
            batch_size, class_num = lm_logits.shape
            # loss = LOSS_fn(lm_logits.view(-1, class_num), sc_labels.view(-1)).view(batch_size)
            # loss = torch.mean(loss)
            predict = torch.max(F.softmax(lm_logits, dim=-1), dim=-1)[1].reshape(batch_size)
            predict_list.extend(predict.tolist())
    return predict_list


def Base_valid(valid_iter, model, args):
    model.eval()
    global SCOREs
    LOSS_fn = nn.CrossEntropyLoss(reduction="none")
    Ss, Gs, SGs = 0, 0, 0
    predicts = []
    golds = []
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
            # predicts.extend(predict)
            # golds.extend(golds)
            S, G, SG = metrics.ner_metrics(predict.cpu(), gold.cpu())
            Ss += S
            Gs += G
            SGs += SG
        elif args.train_type == "sc":
            sc_labels = item["sc_labels"].to(args.device)
            lm_logits = model(input_ids, input_mask)
            batch_size, class_num = lm_logits.shape
            # loss = LOSS_fn(lm_logits.view(-1, class_num), sc_labels.view(-1)).view(batch_size)
            # loss = torch.mean(loss)
            predict = torch.max(lm_logits, dim=-1)[1].reshape(batch_size)
            gold = sc_labels.reshape(batch_size)
            predicts.extend(predict.cpu().tolist())
            golds.extend(gold.cpu().tolist())
            # logging.info(f"predict: {predict.cpu().tolist()}")
            # logging.info(f"gold: {gold.cpu().tolist()}")
    if args.train_type == "ner":
        P = SGs / Ss
        R = SGs / Gs
        F = 2 * P * R / (P + R)
        score_mean = F
    else:
        score_mean = metrics.sc_metrics(predict=predicts, gold=golds)
    logging.info("epoch{} score:{:.4f}".format(args.step, score_mean))
    if len(SCOREs) < 5:
        SCOREs.append(score_mean)
        save(model, args.model_save, score_mean)
        SCOREs = sorted(SCOREs)
    else:
        if score_mean > SCOREs[0]:
            rm(args.model_save, SCOREs[0])
            SCOREs.append(score_mean)
            save(model, args.model_save, score_mean)
            SCOREs = sorted(SCOREs[1:])


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
            # "lr": args.lr * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and not any(nd in n for nd in high_lr)
            ],
            "weight_decay": 0.0,
            # "lr": args.lr * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in high_lr)
            ],
            "weight_decay": 0.0,
            # "lr": args.lr,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    if args.force:
        force_model(model)
    else:
        # train_model(model, 11)
        pass
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logging.info(f"train name: {name}")
    #     else:
    #         logging.info(f"force name: {name}")
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) // args.opt_step, num_training_steps=len(train_iter) * args.epoch // args.opt_step)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) // args.opt_step, num_training_steps=len(train_iter) * args.epoch // args.opt_step)
    mean_loss = 0
    batch_step = 0
    args.step = 1
    LOSS_fn = nn.CrossEntropyLoss(reduction="none")
    for step in range(args.epoch):
        model.train()
        logging.info("Starting Training epoch:{}".format(step+1))
        for idx, item in enumerate(train_iter):
            batch_step += 1
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
            if batch_step % args.opt_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if batch_step % args.eval_step == 0:
                with torch.no_grad():
                    Base_valid(valid_iter, model, args)
                    model.train()
        args.step = step + 1
        mean_loss /= len(train_iter)
        logging.info("Train loss:{:.4f}".format(mean_loss))
        mean_loss = 0
        # if dist.get_rank() == 0:
        with torch.no_grad():
            Base_valid(valid_iter, model, args)
