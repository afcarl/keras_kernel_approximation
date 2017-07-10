import numpy

from utils.prepare_data import load_tiselac
from utils.model_utils import print_train_valid, load_model

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

# Params
d = 10
sz = 23
n_classes = 9

validation_split = .05

# Load training data
X, X_coord, y = load_tiselac(training_set=True, shuffle=True, random_state=0)

# Model definition

# fname_model = "output/models_baseline/mlp.256-128-64.00055-0.2315.weights.hdf5"
fname_model = "output/models_baseline/mlp_rff.256-128-64.00184-acc0.9413.weights.hdf5"
# fname_model = "output/models_baseline/rnn.256.128-64.00344-acc0.9459.weights.hdf5"
model = load_model(fname_model=fname_model, sz=sz, d=d, d_side_info=X_coord.shape[1], n_classes=n_classes)

if fname_model.split("/")[-1].startswith("rnn."):
    X = X.reshape((-1, sz, d))
    X_final = [X, X_coord]
else:
    X_final = numpy.hstack((X, X_coord))

print_train_valid(model=model, X=X_final, y=y, validation_split=validation_split)
