#!/bin/sh
# source ~/.bashrc
# source activate telma
ROOT="$HOME/opt/tiger/NER_SC"
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="0,1"
MODEL="SC"
TYPE="sc"

# Tokenizer and Pre-Train Model
# ------------------------------------------------
# TOKENIZER="hfl/chinese-macbert-large"
# PRETRAIN="hfl/chinese-macbert-large"
# TOKENIZER="hfl/chinese-roberta-wwm-ext-large"
# PRETRAIN="hfl/chinese-roberta-wwm-ext-large"
# TOKENIZER="hfl/chinese-bert-wwm-ext"
# PRETRAIN="hfl/chinese-bert-wwm-ext"
# TOKENIZER="nghuyong/ernie-1.0"
# PRETRAIN="nghuyong/ernie-1.0"
TOKENIZER="bert-base-chinese"
PRETRAIN="bert-base-chinese"
# ------------------------------------------------
# 
LOAD="$ROOT/model/SC/2021_11_08_08_31_score_0.3160.pkl"
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
--batch_size=32 \
--epoch=15 \
--opt_step=2 \
--l_model=768 \
> ../log/Base_sc.log 2>&1 &