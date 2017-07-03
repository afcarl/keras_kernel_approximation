import numpy
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

from mk_rff_learn import model_mk_rff
from prepare_data import ecml17_tiselac_data_preparation
from metrics import f1_score

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def shuffle_data(X, y):
    assert X.shape[0] == y.shape[0]
    indices = numpy.random.permutation(X.shape[0])
    return X[indices], y[indices]

# Params
d = 10
sz = 23
n_classes = 9
rff_dim = 512
feature_sizes = [8, 12, 16]

# Load training data
X = numpy.loadtxt("data_tiselac/training.txt", dtype=numpy.float, delimiter=",")
X /= X.max()
y = numpy.loadtxt("data_tiselac/training_class.txt", delimiter=",").astype(numpy.int) - 1
# /!\ Caution: do reverse transform (+1 for predicted y, divided by max_train for X_test) at test time /!\

# Prepare data
X, y = shuffle_data(X, y)
y_encoded = to_categorical(y)
feats_8_12_16 = ecml17_tiselac_data_preparation(X, d=d, feature_sizes=tuple(feature_sizes), use_time=True)

# Prepare model
dict_dims = {(d * f_sz + 1): sz - f_sz + 1 for f_sz in feature_sizes}
model = model_mk_rff(input_dimensions=dict_dims, embedding_dim=rff_dim, n_classes=n_classes)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])  #, f1_score])

# Just check that weights are shared, not repeated as many times as the number of features in the sets
print("Weights:", [w.shape for w in model.get_weights()])
print("Total number of parameters:", model.count_params())

# Go!
save_model_cb = ModelCheckpoint("models/model_mk_rff." + str(rff_dim) + ".{epoch:03d}-{val_loss:.2f}.weights.hdf5",
                                monitor='val_loss', verbose=False, save_best_only=True, save_weights_only=True,
                                mode='auto', period=1)
early_stopping_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=True, mode='auto')
model.fit(feats_8_12_16, y_encoded, batch_size=128, epochs=10 * 1000, verbose=2, validation_split=0.05,
          callbacks=[save_model_cb, early_stopping_cb])
model.save_weights("models/model_mk_rff.%d.final.weights.hdf5" % rff_dim)
y_pred = model.predict(feats_8_12_16, verbose=False)
eval_model = model.evaluate(feats_8_12_16, y_encoded, verbose=False)
print("Correct classification rate:", eval_model[1])
print("F1-score:", f1_score(y_true=y_encoded, y_pred=y_pred))
