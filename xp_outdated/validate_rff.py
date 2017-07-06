from keras_models.mk_rff_learn import model_mk_rff

from utils.metrics import f1_score
from utils.prepare_data import ecml17_tiselac_data_preparation, load_tiselac

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

# Params
d = 10
sz = 23
n_classes = 9

rff_dim = 256
feature_sizes = [4, 8, 12, 16]

validation_split = .05

# Load training data
X, X_coord, y = load_tiselac(training_set=True, shuffle=True, random_state=0)

# Load model
fname_model = "output/models_rff/4-8-12-16.256.01094-0.3299.weights.hdf5"
dict_dims = {(d * f_sz + 1): sz - f_sz + 1 for f_sz in feature_sizes}
model = model_mk_rff(input_dimensions=dict_dims, embedding_dim=rff_dim, n_classes=n_classes, side_info_dim=2)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.load_weights(fname_model)

for n_valid in [X.shape[0], int(validation_split * X.shape[0])]:
    X_valid = X[-n_valid:]
    X_coord_valid = X_coord[-n_valid:]
    y_valid = y[-n_valid:]

    feats_8_12_16 = ecml17_tiselac_data_preparation(X_valid, d=d, feature_sizes=tuple(feature_sizes), use_time=True)

    y_pred = model.predict(feats_8_12_16 + [X_coord_valid], verbose=False)
    eval_model = model.evaluate(feats_8_12_16 + [X_coord_valid], y_valid, verbose=False)
    if n_valid == X.shape[0]:
        print("Full training set")
    else:
        print("Validation set")
    print("Correct classification rate:", eval_model[1])
    print("F1-score:", f1_score(y_true=y_valid, y_pred=y_pred))
