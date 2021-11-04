import argparse


def Base_config():
    parser = argparse.ArgumentParser()
    # OS config
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_save", type=str)
    parser.add_argument("--model_load", type=str, default=None)
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
    # Predict config
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--predict_type", type=str, default="ner")
    parser.add_argument("--output", type=str)