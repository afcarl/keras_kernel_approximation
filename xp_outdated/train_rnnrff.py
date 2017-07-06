import numpy

from keras_models.mk_rnnrff_learn import model_mk_rnnrff
from utils.prepare_data import load_tiselac, ecml17_tiselac_data_preparation
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
rff_dim = 256
rnn_dim = 32
feature_sizes = [8, 12, 16]

# Load training data
X, X_coord, y = load_tiselac(training_set=True, shuffle=True, random_state=0)
feats_8_12_16 = ecml17_tiselac_data_preparation(X, d=d, feature_sizes=tuple(feature_sizes), make_monodim=False)

# Prepare model
dict_dims = {(f_sz, d): sz - f_sz + 1 for f_sz in feature_sizes}
model = model_mk_rnnrff(input_dimensions=dict_dims, rnn_dim=rnn_dim, rff_dim=rff_dim, n_classes=n_classes)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])  #, f1_score])

# Just check that weights are shared, not repeated as many times as the number of features in the sets
print("Weights:", [w.shape for w in model.get_weights()])
print("Total number of parameters:", model.count_params())

# Fit
basename = "output/models_rnnrff/"
for sz in feature_sizes:
    basename += "%d-" % sz
basename = basename[:-1] + ".%d.%d" % (rnn_dim, rff_dim)
model_fit_and_save(model, basename, X=feats_8_12_16 + [X_coord], y=y)  # TODO

# Go!
print_eval(model, feats_8_12_16 + [X_coord], y)
