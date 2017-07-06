import numpy

from utils.model_utils import model_fit_and_save, print_eval
from utils.prepare_data import load_tiselac, ecml17_tiselac_data_preparation
from keras_models.model_zoo import model_rnn, model_mlp_rff, model_mlp

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

n_units_hidden_layers_rff = [256, 128]
rff_dim = 64

# Load model
fname_model_rff = "output/models_baseline/mlp_rff.256-128.00184-acc0.9413.weights.hdf5"
rff_model = model_mlp_rff(input_shape=(sz * d + 2, ), hidden_layers=n_units_hidden_layers_rff, rff_layer_dim=rff_dim)
rff_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
rff_model.load_weights(fname_model_rff, by_name=True)

rff_features = rff_model.predict(numpy.hstack((X, X_coord)))

# =========
# RNN stuff
# =========
X_rnn = X.reshape((-1, sz, d))

dim_rnn = 256
n_units_hidden_layers_rnn = [128, 64]

# Load model
fname_model_rnn = "output/models_baseline/rnn.256.128-64.00062-acc0.9315.weights.hdf5"
rnn_model = model_rnn(input_shape=(sz, d), hidden_layers=n_units_hidden_layers_rnn, rnn_layer_dim=dim_rnn,
                      input_shape_side_info=(2, ), use_lstm=True)
rnn_model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
rnn_model.load_weights(fname_model_rnn, by_name=True)

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

# Just check that weights are shared, not repeated as many times as the number of features in the sets
print("Weights:", [w.shape for w in ensemble_model.get_weights()])
print("Total number of parameters:", ensemble_model.count_params())

# Fit
basename = "output/models_ensemble/mlp_rnn_rff"  # TODO
model_fit_and_save(ensemble_model, basename, X=ensemble_features, y=y, patience_early_stopping=100, save_acc=True)
print_eval(ensemble_model, ensemble_features, y)

