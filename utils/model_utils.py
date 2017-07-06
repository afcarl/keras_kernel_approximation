from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

from utils.metrics import f1_score

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
    for n_valid in [X.shape[0], int(validation_split * X.shape[0])]:
        if isinstance(X, list):
            X_valid = [Xi[-n_valid:] for Xi in X]
        else:
            X_valid = X[-n_valid:]
        y_valid = y[-n_valid:]

        if n_valid == X.shape[0]:
            print("Full training set")
        else:
            print("Validation set")
        print_eval(model=model, X=X_valid, y=y_valid)


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
