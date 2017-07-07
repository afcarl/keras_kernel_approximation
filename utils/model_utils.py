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
    print("Last saved model: %s" % save_model_cb.last_saved_model)
    return list_callbacks


def print_eval(model, X, y):
    y_pred = model.predict(X, verbose=False)
    eval_model = model.evaluate(X, y, verbose=False)
    print("Correct classification rate:", eval_model[1])
    print("F1-score:", f1_score(y_true=y, y_pred=y_pred))


def print_train_valid(model, X, y, validation_split):
    if isinstance(X, list):
        n = X[0].shape[0]
    else:
        n = X.shape[0]
    for n_valid in [n, int(validation_split * n)]:
        if isinstance(X, list):
            X_valid = [Xi[-n_valid:] for Xi in X]
        else:
            X_valid = X[-n_valid:]
        y_valid = y[-n_valid:]

        if n_valid == n:
            print("Full training set")
        else:
            print("Validation set")
        print_eval(model=model, X=X_valid, y=y_valid)


def load_model(fname_model, sz, d, d_side_info=None, use_lstm=True, n_classes=None):
    """Loads a model from its weight filename (all necessary information should be included in it).

    As for now, this function assumes that default activation function is used for each layer.
    """
    path, basename = os.path.split(fname_model)
    model = None
    if basename.startswith("mlp."):
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
        model = model_mlp_rff(input_shape=(sz * d + d_side_info), hidden_layers=n_units_hidden_layers,
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
