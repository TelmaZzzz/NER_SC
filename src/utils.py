import csv
import torch
import random
import numpy as np
import logging
import metrics
from nlpcda import Similarword, Homophone


def read_from_csv(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, item in enumerate(reader):
            if idx > 0:
                data.append(item)
    return data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed) #为当前GPU设置随机种子


def d2s(dt, time=False):
    if time is False:
        return dt.strftime("%Y_%m_%d")
    else:
        return dt.strftime("%Y_%m_%d_%H_%M")


def debug(name, value):
    logging.debug(f"{name}: {value}")
