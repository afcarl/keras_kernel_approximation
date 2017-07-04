from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.metrics import f1_score

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def model_fit_and_save(model, basename, X, y, max_iter=10000, validation_split=.05, patience_early_stopping=100):
    save_model_cb = ModelCheckpoint(basename + ".{epoch:05d}-{val_loss:.4f}.weights.hdf5",
                                    monitor='val_loss', verbose=False, save_best_only=True, save_weights_only=True,
                                    mode='auto', period=1)
    early_stopping_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_early_stopping, verbose=True,
                                      mode='auto')
    model.fit(X, y, batch_size=128, epochs=max_iter, verbose=2, validation_split=validation_split,
              callbacks=[save_model_cb, early_stopping_cb])
    model.save_weights(basename + ".final.weights.hdf5")


def print_eval(model, X, y):
    y_pred = model.predict(X, verbose=False)
    eval_model = model.evaluate(X, y, verbose=False)
    print("Correct classification rate:", eval_model[1])
    print("F1-score:", f1_score(y_true=y, y_pred=y_pred))