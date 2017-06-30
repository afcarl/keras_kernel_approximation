from sklearn.metrics import f1_score as skl_f1_score
import keras.backend as K

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def f1_score(y_true, y_pred):
    return skl_f1_score(y_true=K.eval(y_true), y_pred=K.eval(y_pred), average="weighted")
