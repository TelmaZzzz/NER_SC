#!/bin/sh
# source ~/.bashrc
# source activate telma
ROOT="$HOME/opt/tiger/NER_SC"
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="0,1"
MODEL="Base"
TYPE="ner"
TOKENIZER="bert-base-chinese"
PRETRAIN="bert-base-chinese"

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
--batch_size=16 \
--epoch=20 \
--opt_step=2 \
--l_model=768 \
> ../log/Base.log 2>&1 &