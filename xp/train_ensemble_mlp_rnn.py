import numpy

from utils.model_utils import model_fit_and_save, print_eval, load_model
from utils.prepare_data import load_tiselac
from keras_models.model_zoo import model_mlp

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

fname_model_rff = "output/models_baseline/mlp_rff.256-128-64.00184-acc0.9413.weights.hdf5"
rff_model = load_model(fname_model_rff, input_shape=(sz * d + 2, ))
rff_features = rff_model.predict(numpy.hstack((X, X_coord)))

# =========
# RNN stuff
# =========
X_rnn = X.reshape((-1, sz, d))

# Load model
fname_model_rnn = "output/models_baseline/rnn.256.128-64.00344-acc0.9459.weights.hdf5"
rnn_model = load_model(fname_model_rnn, input_shape=(sz, d), input_shape_side_info=(2, ))
rnn_features = rnn_model.predict([X_rnn, X_coord])

# # =========
# # MLP stuff
# # =========
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

