from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

from utils.metrics import f1_score
from keras_models.model_zoo import model_mlp, model_mlp_rff, model_rnn

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def model_fit_and_save(model, basename, X, y, max_iter=10000, validation_split=.05, patience_early_stopping=100,
                       save_acc=False):
    if save_acc:
        monitor = "val_acc"
    else:
        monitor = "val_loss"
    save_model_cb = CustomModelCheckpoint(basename + ".{epoch:05d}-{%s:.4f}.weights.hdf5" % monitor,
                                          monitor=monitor, verbose=False, save_best_only=True,
                                          save_weights_only=True, mode='auto', period=1)
    early_stopping_cb = EarlyStopping(monitor=monitor, min_delta=0, patience=patience_early_stopping, verbose=True,
                                      mode='auto')
    list_callbacks = [save_model_cb, early_stopping_cb]
    model.fit(X, y, batch_size=128, epochs=max_iter, verbose=2, validation_split=validation_split,
              callbacks=list_callbacks)
    return save_model_cb.last_saved_model


def print_eval(model, X, y):
    y_pred = model.predict(X, verbose=False)
    eval_model = model.evaluate(X, y, verbose=False)
    print("Correct classification rate:", eval_model[1])
    print("F1-score:", f1_score(y_true=y, y_pred=y_pred))


def print_train_valid_test(model, X, y, validation_split, test_split=None):
    n = n_individuals(X)
    if test_split is None:
        n_test = 0
    else:
        n_test = int(n * test_split)
    n_valid = int((n - n_test) * validation_split)
    n_train = n - n_test - n_valid
    print("Training set")
    print_eval(model=model, X=get_slice(X, 0, n_train), y=y[:n_train])
    if test_split is not None:
        print("Validation set")
        print_eval(model=model, X=get_slice(X, n_train, -n_test), y=y[n_train:-n_test])
        print("Test set")
        print_eval(model=model, X=get_slice(X, -n_test), y=y[-n_test:])
    else:
        print("Validation set")
        print_eval(model=model, X=get_slice(X, n_train), y=y[n_train:])


def n_individuals(X):
    if isinstance(X, list):
        return X[0].shape[0]
    else:
        return X.shape[0]


def get_slice(X, idx_start, idx_end=None):
    if isinstance(X, list):
        return [get_slice(Xi, idx_start, idx_end) for Xi in X]
    else:
        if idx_end is None:
            return X[idx_start:]
        else:
            return X[idx_start:idx_end]


def load_model(fname_model, sz, d, d_side_info=0, use_lstm=True, n_classes=None):
    """Loads a model from its weight filename (all necessary information should be included in it).

    As for now, this function assumes that default activation function is used for each layer.
    """
    path, basename = os.path.split(fname_model)
    model = None
    if basename.startswith("mlp.") or basename.startswith("mlp_rnn_rff."):
        # MLP model
        s_layer_sizes = basename.split(".")[1]
        n_units_hidden_layers = [int(s) for s in s_layer_sizes.split("-")]
        model = model_mlp(input_shape=(sz * d + d_side_info, ), hidden_layers=n_units_hidden_layers,
                          n_classes=n_classes)
    elif basename.startswith("rnn."):
        # RNN model
        s_rnn_dim, s_layer_sizes = basename.split(".")[1:3]
        dim_rnn = int(s_rnn_dim)
        n_units_hidden_layers = [int(s) for s in s_layer_sizes.split("-")]
        model = model_rnn(input_shape=(sz, d), hidden_layers=n_units_hidden_layers, rnn_layer_dim=dim_rnn,
                          input_shape_side_info=(d_side_info, ), n_classes=n_classes, use_lstm=use_lstm)
    elif basename.startswith("mlp_rff."):
        # MLP-RFF model
        s_layer_sizes = basename.split(".")[1]
        n_units_hidden_layers = [int(s) for s in s_layer_sizes.split("-")]
        rff_dim = n_units_hidden_layers[-1]
        n_units_hidden_layers.pop()
        model = model_mlp_rff(input_shape=(sz * d + d_side_info, ), hidden_layers=n_units_hidden_layers,
                              rff_layer_dim=rff_dim, n_classes=n_classes)
    else:
        raise ValueError("Cannot interpret file name %s" % basename)
    if model is not None:
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        model.load_weights(fname_model, by_name=True)
        return model


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        ModelCheckpoint.__init__(self, filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only,
                                 save_weights_only=save_weights_only, mode=mode, period=period)
        self.last_saved_model = ""

    def on_epoch_end(self, epoch, logs=None):
        if self.save_best_only and self.monitor_op(logs.get(self.monitor), self.best):
            model_to_be_removed = self.last_saved_model
            self.last_saved_model = self.filepath.format(epoch=epoch, **logs)
        else:
            model_to_be_removed = None
        ModelCheckpoint.on_epoch_end(self, epoch=epoch, logs=logs)
        if model_to_be_removed is not None and model_to_be_removed != "":
            os.remove(model_to_be_removed)
