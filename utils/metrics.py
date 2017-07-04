from sklearn.metrics import f1_score as skl_f1_score

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def f1_score(y_true, y_pred):
    return skl_f1_score(y_true=y_true.argmax(axis=1), y_pred=y_pred.argmax(axis=1), average="weighted")
