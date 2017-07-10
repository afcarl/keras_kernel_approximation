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
rff_model = load_model(fname_model_rff, sz=sz, d=d, d_side_info=2)
rff_features = rff_model.predict(numpy.hstack((X, X_coord)))

# =========
# RNN stuff
# =========
X_rnn = X.reshape((-1, sz, d))
fname_model_rnn = "output/models_baseline/rnn.256.128-64.00422-0.9422.weights.hdf5"
rnn_model = load_model(fname_model_rnn, sz=sz, d=d, d_side_info=2)
rnn_features = rnn_model.predict([X_rnn, X_coord])

# =========
# MLP stuff
# =========
fname_model_mlp = "output/models_baseline/mlp.256-128-64.00231-0.9413.weights.hdf5"
mlp_model = load_model(fname_model_mlp, sz=sz, d=d, d_side_info=2)
mlp_features = mlp_model.predict(numpy.hstack((X, X_coord)))

# ==============
# Ensemble stuff
# ==============
ensemble_features = numpy.hstack((mlp_features, rnn_features, rff_features))

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

model_fit_and_save(ensemble_model, basename, X=ensemble_features, y=y, patience_early_stopping=100, save_acc=True)
print_eval(ensemble_model, ensemble_features, y)

