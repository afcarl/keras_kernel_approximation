from utils.prepare_data import load_tiselac
from utils.model_utils import print_train_valid
from keras_models.model_zoo import model_rnn

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

# Params
d = 10
sz = 23
n_classes = 9

dim_rnn = 256
n_units_hidden_layers = [128, 64]

validation_split = .05

# Load training data
X, X_coord, y = load_tiselac(training_set=True, shuffle=True, random_state=0)
X = X.reshape((-1, sz, d))

# Model definition
fname_model = "output/models_baseline/rnn.256.128-64.00134-0.2254.weights.hdf5"
model = model_rnn(input_shape=(sz, d), hidden_layers=n_units_hidden_layers, rnn_layer_dim=dim_rnn,
                  input_shape_side_info=(2, ), n_classes=n_classes, use_lstm=True)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.load_weights(fname_model)

print_train_valid(model=model, X=[X, X_coord], y=y, validation_split=validation_split)
