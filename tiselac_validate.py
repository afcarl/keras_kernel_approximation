import numpy
from keras.utils import to_categorical

from prepare_data import ecml17_tiselac_data_preparation
from metrics import f1_score
from mk_rff_learn import model_mk_rff


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
y_train = numpy.loadtxt("data_tiselac/training_class.txt", delimiter=",").astype(numpy.int)

y_encoded = to_categorical(y_train)
feats_8_12_16 = ecml17_tiselac_data_preparation(X_train / xmax_train, d=d, feature_sizes=tuple(feature_sizes),
                                                use_time=True)

# Load model
fname_model = "models/model_mk_rff.%d.069-0.53.weights.hdf5" % rff_dim  # TODO
dict_dims = {(d * f_sz + 1): sz - f_sz + 1 for f_sz in feature_sizes}
model = model_mk_rff(input_dimensions=dict_dims, embedding_dim=rff_dim, n_classes=n_classes)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.load_weights(fname_model)

y_pred = model.predict(feats_8_12_16, verbose=False)
eval_model = model.evaluate(feats_8_12_16, y_encoded, verbose=False)
print("Correct classification rate:", eval_model[1])
print("F1-score:", f1_score(y_true=y_encoded, y_pred=y_pred))
