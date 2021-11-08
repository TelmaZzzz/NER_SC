#!/bin/sh
# source ~/.bashrc
# source activate telma
ROOT="$HOME/opt/tiger/NER_SC"
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="0,1"
MODEL="SC"
TYPE="sc"
# TOKENIZER="hfl/chinese-roberta-wwm-ext-large"
TOKENIZER="bert-base-chinese"
# PRETRAIN="bert-base-chinese"
# PRETRAIN="hfl/chinese-roberta-wwm-ext-large"

# TRAIN_PATH="$ROOT/data/train_data_public.csv"
TEST_PATH="$ROOT/data/test_public.csv"

# python -m torch.distributed.launch --nproc_per_node 2 ../src/Base.py \
python ../src/Base.py \
--predict \
--test_path="$TEST_PATH" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--ner_model_load="$ROOT/model/NER/2021_11_08_00_23_score_0.7484.pkl" \
--sc_model_load="$ROOT/model/SC/2021_11_07_17_45_score_0.4921.pkl" \
--batch_size=16 \
--output_path="$ROOT/output/ans_3.csv" \
> ../log/Base_predict.log 2>&1 &