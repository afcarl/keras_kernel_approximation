from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_circles
import numpy

from layers import RFFLayer
from metrics import f1_score

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def model_rff(input_dim, embedding_dim, n_classes):
    model = Sequential()
    model.add(RFFLayer(units=embedding_dim, input_dim=input_dim))
    model.add(Dense(units=n_classes, activation="softmax"))
    return model


if __name__ == "__main__":
    # Data
    n_samples = 10000
    d = 2
    n_classes = 2
    X, y = make_circles(n_samples=n_samples, random_state=0, noise=.1, factor=.5)
    embedding_dim = 128
    y_encoded = numpy.empty((n_samples, n_classes), dtype=numpy.int32)
    y_encoded[:, 0] = 1 - y
    y_encoded[:, 1] = y

    # Model
    model = model_rff(input_dim=d, embedding_dim=embedding_dim, n_classes=n_classes)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy", f1_score])

    # Fit & predict
    model.fit(X, y_encoded, batch_size=128, epochs=50, verbose=True)

    eval_model = model.evaluate(X, y_encoded, verbose=False)
    print("Correct classification rate:", eval_model[1])
    print("F1-score:", eval_model[2])

    del model
