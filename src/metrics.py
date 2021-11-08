from sklearn.metrics import f1_score, cohen_kappa_score


def get_tuple(list):
    tuples = []
    l, r = 0, 0
    for i in range(1, len(list)):
        if list[i] == list[i-1]:
            r = i
        else:
            if list[r] != 0:
                tuples.append((l, r, list[r]))
                l = i
                r = i
    if list[r] != 0:
        tuples.append((l, r, list[r]))
    return tuples


def get_same(S, G):
    ans = 0
    for i in S:
        for j in G:
            if i == j:
                ans += 1
    return ans


def ner_metrics(predict, gold):
    S = get_tuple(predict)
    G = get_tuple(gold)
    SG = get_same(S, G)
    return len(S), len(G), SG


def sc_metrics(predict, gold):
    return cohen_kappa_score(gold, predict)

