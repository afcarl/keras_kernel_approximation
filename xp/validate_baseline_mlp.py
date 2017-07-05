from keras.layers import Dense, Input
from keras.models import Model
import numpy

from utils.metrics import f1_score
from utils.prepare_data import load_tiselac

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

# Params
d = 10
sz = 23
n_classes = 9

n_units_hidden_layers = [256, 128, 64]

validation_split = .05

# Load training data
X, X_coord, y = load_tiselac(training_set=True, shuffle=True, random_state=0)

# Load model
fname_model = "output/models_baseline/mlp.256-128-64.00055-0.2315.weights.hdf5"

# Model definition
input = Input(shape=(sz * d + 2, ))
input_layer = input
for n_units in n_units_hidden_layers:
    output_layer = Dense(units=n_units, activation="tanh")(input_layer)
    input_layer = output_layer
preds = Dense(units=n_classes, activation="softmax")(input_layer)
model = Model(inputs=input, outputs=preds)

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.load_weights(fname_model)

for n_valid in [X.shape[0], int(validation_split * X.shape[0])]:
    X_valid = X[-n_valid:]
    X_coord_valid = X_coord[-n_valid:]
    y_valid = y[-n_valid:]

    y_pred = model.predict(numpy.hstack((X_valid, X_coord_valid)), verbose=False)
    eval_model = model.evaluate(numpy.hstack((X_valid, X_coord_valid)), y_valid, verbose=False)
    if n_valid == X.shape[0]:
        print("Full training set")
    else:
        print("Validation set")
    print("Correct classification rate:", eval_model[1])
    print("F1-score:", f1_score(y_true=y_valid, y_pred=y_pred))
