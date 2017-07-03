import numpy
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Dense, Input

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
n_units1 = 128
n_units2 = 64

# Load training data
X = numpy.loadtxt("data_tiselac/training.txt", dtype=numpy.float, delimiter=",")
X /= X.max()
y = numpy.loadtxt("data_tiselac/training_class.txt", delimiter=",").astype(numpy.int) - 1
# /!\ Caution: do reverse transform (+1 for predicted y, divided by max_train for X_test) at test time /!\

# Prepare data
X, y = shuffle_data(X, y)
y_encoded = to_categorical(y)

# Model definition
input = Input(shape=(sz * d, ))
layer1 = Dense(units=n_units1, activation="tanh")(input)
layer2 = Dense(units=n_units2, activation="tanh")(layer1)
preds = Dense(units=n_classes, activation="softmax")(layer2)
model = Model(inputs=input, outputs=preds)


model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])  #, f1_score])

# Just check that weights are shared, not repeated as many times as the number of features in the sets
print("Weights:", [w.shape for w in model.get_weights()])
print("Total number of parameters:", model.count_params())

# Go!
save_model_cb = ModelCheckpoint("models_baseline/model_mlp_baseline.%d-%d.{epoch:03d}-{val_loss:.2f}.weights.hdf5" %
                                (n_units1, n_units2), monitor='val_loss', verbose=False, save_best_only=True,
                                save_weights_only=True, mode='auto', period=1)
early_stopping_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=True, mode='auto')
model.fit(X, y_encoded, batch_size=128, epochs=10 * 1000, verbose=2, validation_split=0.05,
          callbacks=[save_model_cb, early_stopping_cb])
model.save_weights("models_baseline/model_mlp_baseline.%d-%d.final.weights.hdf5" % (n_units1, n_units2))
y_pred = model.predict(X, verbose=False)
eval_model = model.evaluate(X, y_encoded, verbose=False)
print("Correct classification rate:", eval_model[1])
print("F1-score:", f1_score(y_true=y_encoded, y_pred=y_pred))
