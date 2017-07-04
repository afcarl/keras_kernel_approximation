import numpy
from keras.layers import Dense, Input, Conv1D, SimpleRNN
from keras.models import Model

from utils.prepare_data import load_tiselac
from utils.model_utils import model_fit_and_save, print_eval

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def shuffle_data(X, y):
    assert X.shape[0] == y.shape[0]
    indices = numpy.random.permutation(X.shape[0])
    return X[indices], y[indices]

# Params
d = 10
sz = 23
n_classes = 9

convsz = 8
n_filters_conv = 10
rnn_dim = 512

# Load training data
X, X_coord, y = load_tiselac(training_set=True, shuffle=True, random_state=0)

# Model definition
input = Input(shape=(sz, d))
conv_layer = Conv1D(filters=n_filters_conv, kernel_size=convsz)(input)
rnn_layer = SimpleRNN(units=rnn_dim)(conv_layer)
preds = Dense(units=n_classes, activation="softmax")(rnn_layer)
model = Model(inputs=input, outputs=preds)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Just check that weights are shared, not repeated as many times as the number of features in the sets
print("Weights:", [w.shape for w in model.get_weights()])
print("Total number of parameters:", model.count_params())

# Fit
basename = "output/models_baseline/crnn.%d-%d-%d" % (convsz, n_filters_conv, rnn_dim)
model_fit_and_save(model, basename, X=X, y=y)

# Go!
print_eval(model, X, y)
