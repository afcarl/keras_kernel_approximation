import numpy
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy

from mk_rff_learn import model_mk_rff
from prepare_data import ecml17_tiselac_data_preparation

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

# Params
n_ts = 82230
d = 10
sz = 23
n_classes = 9
rff_dim = 128
feature_sizes = [8, 12, 16]

# TODO: change for load from disk
X = numpy.empty((n_ts, d * sz))
y = numpy.random.randint(low=0, high=n_classes, size=n_ts)
# END TODO

# Prepare data
y_encoded = to_categorical(y, num_classes=n_classes)
feats_8_12_16 = ecml17_tiselac_data_preparation(X, d=d, feature_sizes=tuple(feature_sizes), use_time=True)

# Prepare model
dict_dims = {(d * f_sz + 1): sz - f_sz + 1 for f_sz in feature_sizes}
model = model_mk_rff(input_dimensions=dict_dims, embedding_dim=rff_dim, n_classes=n_classes)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=[categorical_accuracy])

# Just check that weights are shared, not repeated as many times as the number of features in the sets
print("Weights:", [w.shape for w in model.get_weights()])
print("Total number of parameters:", model.count_params())

# Go!
model.fit(feats_8_12_16, y_encoded, batch_size=128, epochs=1, verbose=True)
y_pred = model.predict(feats_8_12_16, verbose=False)
print("Correct classification rate: ", model.evaluate(feats_8_12_16, y_encoded, verbose=False)[1])
