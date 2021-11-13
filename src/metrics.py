from sklearn.metrics import f1_score, cohen_kappa_score
import logging


def get_tuple(list):
    tuples = []
    l, r = 0, 0
    for i in range(1, len(list)):
        if list[i] == list[l] + 1 and list[l] % 2 == 1:
            r = i
        else:
            if list[l] % 2 == 1:
                tuples.append((l, r, list[l]))
            l = i
            r = i
    if list[l] % 2 == 1:
        tuples.append((l, r, list[l]))
    return tuples


def get_same(S, G):
    ans = 0
    for i in S:
        for j in G:
            if i == j:
                ans += 1
    return ans


def ner_metrics(predict, gold):
    # logging.info(f"predict: {predict}")
    # logging.info(f"gold: {gold}")
    S = get_tuple(predict)
    G = get_tuple(gold)
    SG = get_same(S, G)
    # logging.info(f"S: {S}. G: {G}. SG: {SG}")
    return len(S), len(G), SG


def sc_metrics(predict, gold):
    acc, pp, np = 0, 0, 0
    for p, g in zip(predict, gold):
        if p == g:
            acc += 1
            if p == 0:
                np += 1
            elif p == 1:
                pp += 1
    logging.info(f"acc: {acc}, len: {len(predict)}, pp: {pp}, np: {np}, acc/len: {acc / len(predict)}")
    return cohen_kappa_score(gold, predict)

