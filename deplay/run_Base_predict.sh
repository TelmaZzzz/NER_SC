#!/bin/sh
# source ~/.bashrc
# source activate telma
ROOT="$HOME/opt/tiger/NER_SC"
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="0,1"
MODEL="SC"
TYPE="sc"
# TOKENIZER="hfl/chinese-roberta-wwm-ext-large"
# TOKENIZER_SC="bert-base-chinese"
TOKENIZER_SC="nghuyong/ernie-1.0"
TOKENIZER_NER="hfl/chinese-roberta-wwm-ext-large"
# PRETRAIN="bert-base-chinese"
# PRETRAIN="hfl/chinese-roberta-wwm-ext-large"

# TRAIN_PATH="$ROOT/data/train_data_public.csv"
TEST_PATH="$ROOT/data/test_public.csv"
# SC_MODEL="$ROOT/model/SC/2021_11_09_22_39_score_0.5183.pkl"           # bert-base 3层unforce
# SC_MODEL="$ROOT/model/SC/2021_11_08_22_55_score_0.5190.pkl"           # bert-base 2轮
SC_MODEL="$ROOT/model/SC/2021_11_11_13_22_score_0.5147.pkl"           # ernie-1.0 2层unforce seed 959794 ***
# SC_MODEL="$ROOT/model/SC/2021_11_11_18_14_score_0.6984.pkl"           # ernie-1.0 2层unforce seed 794959 xxx
# SC_MODEL="$ROOT/model/SC/2021_11_11_19_19_score_0.7058.pkl"           # ernie-1.0 2层unforce seed 19980917 ___
# SC_MODEL="$ROOT/model/SC/2021_11_12_14_05_score_0.5051.pkl"           # ernie-1.0 2层unforce seed 19990711 ___
# SC_MODEL="$ROOT/model/SC/2021_11_12_15_44_score_0.6452.pkl"           # ernie-1.0 2层unforce seed 959 ___
# -----------------------------------------------------------
# NER_MODEL="$ROOT/model/NER/2021_11_09_11_42_score_0.8364.pkl"         # roberta
NER_MODEL="$ROOT/model/NER/2021_11_12_00_28_score_0.8367.pkl"         # roberta+crf ***
# NER_MODEL="$ROOT/model/NER/2021_11_12_00_28_score_0.8365.pkl"           # roberta+crf 
# NER_MODEL="$ROOT/model/NER/2021_11_12_16_40_score_0.8360.pkl"           # roberta+crf seed 794520
# -----------------------------------------------------------
# python -m torch.distributed.launch --nproc_per_node 2 ../src/Base.py \
python ../src/Base.py \
--predict \
--test_path="$TEST_PATH" \
--tokenizer_sc_path="$TOKENIZER_SC" \
--tokenizer_ner_path="$TOKENIZER_NER" \
--ner_model_load="$NER_MODEL" \
--sc_model_load="$SC_MODEL" \
--batch_size=16 \
--output_path="$ROOT/output/res_4.csv" \
--ensemble \
--model1="$ROOT/model/SC/2021_11_11_13_22_score_0.5147.pkl" \
--model2="$ROOT/model/SC/2021_11_11_19_19_score_0.7058.pkl" \
--model3="$ROOT/model/SC/2021_11_12_15_44_score_0.6452.pkl" \
--crf \
> ../log/Base_predict.log 2>&1 & 