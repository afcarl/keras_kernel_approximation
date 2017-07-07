import numpy

from utils.prepare_data import load_tiselac
from keras_models.model_zoo import model_mlp_rff, model_mlp, model_rnn
from utils.model_utils import print_train_valid, load_model

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

# Params
d = 10
sz = 23
n_classes = 9

validation_split = .05

# Load training data
X, X_coord, y = load_tiselac(training_set=True, shuffle=True, random_state=0)

# =========
# RFF stuff
# =========

fname_model_rff = "output/models_baseline/mlp_rff.256-128.00184-acc0.9413.weights.hdf5"
rff_model = load_model(fname_model_rff, input_shape=(sz * d + 2, ))
rff_features = rff_model.predict(numpy.hstack((X, X_coord)))

# =========
# RNN stuff
# =========
X_rnn = X.reshape((-1, sz, d))

# Load model
fname_model_rnn = "output/models_baseline/rnn.256.128-64.00344-acc0.9459.weights.hdf5"
rnn_model = load_model(fname_model_rff, input_shape=(sz, d), input_shape_side_info=(2, ))
rnn_features = rnn_model.predict([X_rnn, X_coord])

# # =========
# # MLP stuff
# # =========
# X_rnn = X.reshape((-1, sz, d))
#
# n_units_hidden_layers_mlp = [256, 128, 64]
#
# # Load model
# fname_model_mlp = "output/models_baseline/mlp.256-128-64.00068-0.2283.weights.hdf5"
#
# # Model definition
# mlp_model = model_mlp(input_shape=(sz * d + 2, ), hidden_layers=n_units_hidden_layers, n_classes=n_classes,
# activation="relu")
#
# mlp_model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
# mlp_model.load_weights(fname_model_mlp, by_name=True)
#
# mlp_features = mlp_model.predict(numpy.hstack((X, X_coord)))

# ==============
# Ensemble stuff
# ==============
#ensemble_features = numpy.hstack((mlp_features, rnn_features, rff_features))
ensemble_features = numpy.hstack((rnn_features, rff_features))

n_units_hidden_layers_ensemble = [64, 32]

# Model definition
ensemble_model = model_mlp(input_shape=(ensemble_features.shape[1], ), hidden_layers=n_units_hidden_layers_ensemble,
                           n_classes=n_classes)
ensemble_model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
fname_model_ensemble = "output/models_ensemble/mlp_rnn_rff.00031-0.4527.weights.hdf5"
ensemble_model.load_weights(fname_model_ensemble)

print_train_valid(model=ensemble_model, X=ensemble_features, y=y, validation_split=validation_split)
