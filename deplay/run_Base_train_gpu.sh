#!/bin/sh
source ~/.bashrc
source activate telma
ROOT="$HOME/opt/tiger/NER_SC"
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="0,1"
MODEL="NER"
TYPE="ner"

# Tokenizer and Pre-Train Model
# ------------------------------------------------
# TOKENIZER="hfl/chinese-macbert-large"
# PRETRAIN="hfl/chinese-macbert-large"
TOKENIZER="hfl/chinese-roberta-wwm-ext-large"
PRETRAIN="hfl/chinese-roberta-wwm-ext-large"
# TOKENIZER="hfl/chinese-bert-wwm-ext"
# PRETRAIN="hfl/chinese-bert-wwm-ext"
# TOKENIZER="nghuyong/ernie-1.0"
# PRETRAIN="nghuyong/ernie-1.0"
# TOKENIZER="bert-base-chinese"
# PRETRAIN="bert-base-chinese"
# PRETRAIN="ckiplab/bert-base-chinese-ner"
# ------------------------------------------------
# 
LOAD="$ROOT/model/NER/2021_11_11_18_15_score_0.7801.pkl"
TRAIN_PATH="$ROOT/data/train_data_public.csv"

# python -m torch.distributed.launch --nproc_per_node 2 ../src/Base.py \
python ../src/Base.py \
--train \
--train_type="$TYPE" \
--train_path="$TRAIN_PATH" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$ROOT/model/$MODEL" \
--lr=0.00003 \
--crf \
--batch_size=1 \
--epoch=25 \
--opt_step=64 \
--eval_step=3000 \
--l_model=1024 \
# > ../log/Base_sc.log 2>&1 &