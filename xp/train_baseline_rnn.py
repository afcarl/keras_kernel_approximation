import numpy
from keras.layers import Dense, Input, SimpleRNN, concatenate
from keras.models import Model

from utils.prepare_data import load_tiselac
from utils.model_utils import model_fit_and_save, print_eval

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

# Params
d = 10
sz = 23
n_classes = 9

dim_rnn = 256
n_units_hidden_layers = [128, 64]

# Load training data
X, X_coord, y = load_tiselac(training_set=True, shuffle=True, random_state=0)
X = X.reshape((-1, sz, d))

# Model definition
input = Input(shape=(sz, d))
input_side_info = Input(shape=(2, ))
rnn_layer = SimpleRNN(units=dim_rnn)(input)
input_layer = concatenate([rnn_layer, input_side_info])
for n_units in n_units_hidden_layers:
    output_layer = Dense(units=n_units, activation="tanh")(input_layer)
    input_layer = output_layer
preds = Dense(units=n_classes, activation="softmax")(input_layer)
model = Model(inputs=[input, input_side_info], outputs=preds)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Just check that weights are shared, not repeated as many times as the number of features in the sets
print("Weights:", [w.shape for w in model.get_weights()])
print("Total number of parameters:", model.count_params())

# Fit
basename = "output/models_baseline/rnn.%d." % dim_rnn
for n_units in n_units_hidden_layers:
    basename += "%d-" % n_units
basename = basename[:-1]
model_fit_and_save(model, basename, X=[X, X_coord], y=y, patience_early_stopping=100)

# Go!
print_eval(model, numpy.hstack((X, X_coord)), y)
