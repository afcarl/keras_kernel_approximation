from keras_models.mk_rff_learn import model_mk_rff
from utils.prepare_data import ecml17_tiselac_data_preparation, load_tiselac
from utils.model_utils import model_fit_and_save, print_eval

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

# Params
d = 10
sz = 23
n_classes = 9

rff_dim = 256
feature_sizes = [4, 8, 12, 16]

# Load training data
X, X_coord, y = load_tiselac(training_set=True, shuffle=True, random_state=0)
feats_8_12_16 = ecml17_tiselac_data_preparation(X, d=d, feature_sizes=tuple(feature_sizes), use_time=True)

# Prepare model
dict_dims = {(d * f_sz + 1): sz - f_sz + 1 for f_sz in feature_sizes}
model = model_mk_rff(input_dimensions=dict_dims, embedding_dim=rff_dim, n_classes=n_classes, side_info_dim=2)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Just check that weights are shared, not repeated as many times as the number of features in the sets
print("Weights:", [w.shape for w in model.get_weights()])
print("Total number of parameters:", model.count_params())

# Fit
basename = "output/models_rff/"
for sz in feature_sizes:
    basename += "%d-" % sz
basename = basename[:-1] + ".%d" % rff_dim
model_fit_and_save(model, basename, X=feats_8_12_16 + [X_coord], y=y)

# Go!
print_eval(model, feats_8_12_16 + [X_coord], y)
