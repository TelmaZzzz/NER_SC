import argparse


def Base_config():
    parser = argparse.ArgumentParser()
    # OS config
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_save", type=str)
    parser.add_argument("--model_load", type=str, default=None)
    parser.add_argument("--seed", type=int, default=959794)
    # Train config
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_type", type=str, default="ner")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--fix_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--pretrain_path", type=str)
    parser.add_argument("--opt_step", type=int, default=1)
    parser.add_argument("--l_model", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--eval_step", type=int, default=50)
    parser.add_argument("--unforce", type=int, default=-1)
    parser.add_argument("--crf", action="store_true")
    # Predict config
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--ner_model_load", type=str)
    parser.add_argument("--sc_model_load", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--tokenizer_sc_path", type=str)
    parser.add_argument("--tokenizer_ner_path", type=str)
    parser.add_argument("--model1", type=str)
    parser.add_argument("--model2", type=str)
    parser.add_argument("--model3", type=str)
    parser.add_argument("--model4", type=str)
    parser.add_argument("--model5", type=str)
    parser.add_argument("--model_1", type=str)
    parser.add_argument("--model_2", type=str)
    parser.add_argument("--model_3", type=str)
    parser.add_argument("--ensemble", action="store_true")

    args = parser.parse_args()
    return args