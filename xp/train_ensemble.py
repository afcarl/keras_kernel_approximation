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
X, X_coord, y, X_test, X_coord_test, y_test = load_tiselac(training_set=True, test_split=.05, shuffle=True,
                                                           random_state=0)

# =========
# RFF stuff
# =========
fname_model_rff = "output/models_baseline/mlp_rff.256-128-64.00316-0.9456.weights.hdf5"
rff_model = load_model(fname_model_rff, sz=sz, d=d, d_side_info=2)
rff_features = rff_model.predict(numpy.hstack((X, X_coord)))
rff_features_test = rff_model.predict(numpy.hstack((X_test, X_coord_test)))

# =========
# RNN stuff
# =========
X_rnn = X.reshape((-1, sz, d))
X_rnn_test = X_test.reshape((-1, sz, d))
fname_model_rnn = "output/models_baseline/rnn.256.128-64.00409-0.9404.weights.hdf5"
rnn_model = load_model(fname_model_rnn, sz=sz, d=d, d_side_info=2)
rnn_features = rnn_model.predict([X_rnn, X_coord])
rnn_features_test = rnn_model.predict([X_rnn_test, X_coord_test])

# =========
# MLP stuff
# =========
fname_model_mlp = "output/models_baseline/mlp.256-128-64.00258-0.9440.weights.hdf5"
mlp_model = load_model(fname_model_mlp, sz=sz, d=d, d_side_info=2)
mlp_features = mlp_model.predict(numpy.hstack((X, X_coord)))
mlp_features_test = mlp_model.predict(numpy.hstack((X_test, X_coord_test)))

# ==============
# Ensemble stuff
# ==============
ensemble_features = numpy.hstack((mlp_features, rnn_features, rff_features))
ensemble_features_test = numpy.hstack((mlp_features_test, rnn_features_test, rff_features_test))

n_units_hidden_layers_ensemble = [64, 32]

# Model definition
ensemble_model = model_mlp(input_shape=(ensemble_features.shape[1], ), hidden_layers=n_units_hidden_layers_ensemble,
                           n_classes=n_classes)
ensemble_model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Just check that weights are shared, not repeated as many times as the number of features in the sets
print("Weights:", [w.shape for w in ensemble_model.get_weights()])
print("Total number of parameters:", ensemble_model.count_params())

# Fit
basename = "output/models_ensemble/mlp_rnn_rff."
for units in n_units_hidden_layers_ensemble:
    basename += "%d-" % units
basename = basename[:-1] + "."
short_rnn = fname_model_rnn.split("/")[-1]
short_rnn = short_rnn[:short_rnn.rfind(".weights")]
short_mlp = fname_model_mlp.split("/")[-1]
short_mlp = short_mlp[:short_mlp.rfind(".weights")]
short_rff = fname_model_rff.split("/")[-1]
short_rff = short_rff[:short_rff.rfind(".weights")]
basename += short_rnn + "." + short_mlp + "." + short_rff

fname_weights = model_fit_and_save(ensemble_model, basename, X=ensemble_features, y=y, patience_early_stopping=100,
                                   save_acc=True, validation_split=0.1)
ensemble_model.load_weights(fname_weights)

# Go!
print_eval(ensemble_model, ensemble_features_test, y_test)
