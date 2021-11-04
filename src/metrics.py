from sklearn.metrics import f1_score, cohen_kappa_score

def ner_metrics(predict, gold):
    return f1_score(y_true=gold.tolist(), y_pred=predict.tolist(), average="weighted")


def sc_metrics(predict, gold):
    return cohen_kappa_score(gold.tolist(), predict.tolist())

