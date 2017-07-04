import numpy

from keras_models.mk_rff_learn import model_mk_rff
from utils.prepare_data import ecml17_tiselac_data_preparation


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

# Params
d = 10
sz = 23
n_classes = 9
rff_dim = 512
feature_sizes = [8, 12, 16]

# Load training data
X_train = numpy.loadtxt("data_tiselac/training.txt", dtype=numpy.float, delimiter=",")
xmax_train = X_train.max()

#Load test data
X_test = numpy.loadtxt("data_tiselac/test.txt", dtype=numpy.float, delimiter=",")
X_test /= xmax_train

feats_8_12_16 = ecml17_tiselac_data_preparation(X_test, d=d, feature_sizes=tuple(feature_sizes), use_time=True)

# Load model
fname_model = "models/model_mk_rff.%d.069-0.53.weights.hdf5" % rff_dim  # TODO
dict_dims = {(d * f_sz + 1): sz - f_sz + 1 for f_sz in feature_sizes}
model = model_mk_rff(input_dimensions=dict_dims, embedding_dim=rff_dim, n_classes=n_classes)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.load_weights(fname_model)

y_pred = model.predict(feats_8_12_16, verbose=False)

numpy.savetxt("pred/model_mk_rff.%d.probas.txt" % rff_dim, y_pred)
numpy.savetxt("pred/model_mk_rff.%d.txt" % rff_dim, y_pred.argmax(axis=1) + 1)

