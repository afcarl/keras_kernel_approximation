import numpy
from keras.layers import Dense, Input
from keras.layers.merge import average, concatenate
from keras.models import Model
from sklearn.datasets import make_circles

from utils.layers import RFFLayer

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def model_mk_rff(input_dimensions, embedding_dim, n_classes, side_info_dim=0):
    """Build a match kernel-like model that computes the average of RFF feature for a set and then classifies it using
    Logistic Regression (since the embedding space is supposed to be a good place for linear separability of the sets).

    Parameters
    ----------
    input_dimensions : dict
        Dictionary of dimensions of the input features: {dim0: n_features0, dim1: n_features1, ...}
    embedding_dim : int
        Dimension of the RFF output space (approximation of the RKHS associated to the best kernel)
    n_classes : int
        Number of classes for the classification problem

    Returns
    -------
    Model
        The full model (including the Logistic Regression step), but not compiled

    Examples
    --------
    >>> m = model_mk_rff(input_dimensions={2: 5}, embedding_dim=10, n_classes=2)
    >>> m.count_params()
    52
    """
    inputs = []
    for d in sorted(input_dimensions.keys()):
        n_features = input_dimensions[d]
        inputs.extend([Input(shape=(d, )) for _ in range(n_features)])
    if len(inputs) == 1:
        rff_layer = RFFLayer(units=embedding_dim)
        concatenated_avg_rffs = rff_layer(inputs[0])
    elif len(input_dimensions) == 1:
        rff_layer = RFFLayer(units=embedding_dim)
        rffs = [rff_layer(input_feature) for input_feature in inputs]
        concatenated_avg_rffs = average(rffs)
    else:
        avg_rffs = []
        idx0 = 0
        rff_layers = {}
        for d in sorted(input_dimensions.keys()):
            n_features = input_dimensions[d]

            rff_layers[d] = RFFLayer(units=embedding_dim)
            rffs = [rff_layers[d](input_feature) for input_feature in inputs[idx0:idx0+n_features]]

            avg_rffs.append(average(rffs))
            idx0 += n_features
        concatenated_avg_rffs = concatenate(avg_rffs)
    if side_info_dim > 0:
        side_info_input = Input(shape=(side_info_dim, ))
        concatenated_pred_input = concatenate([concatenated_avg_rffs, side_info_input])
        inputs.append(side_info_input)
    else:
        concatenated_pred_input = concatenated_avg_rffs
    predictions = Dense(units=n_classes, activation="softmax")(concatenated_pred_input)

    return Model(inputs=inputs, outputs=predictions)


if __name__ == "__main__":
    # Data
    n_samples = 100000
    n_sets = 10000
    d = 2
    n_classes = 2
    n_per_set = 3
    X, y = make_circles(n_samples=n_samples, random_state=0, noise=.1, factor=.5)
    embedding_dim = 128
    y_encoded = numpy.zeros((n_sets, n_classes), dtype=numpy.int32)
    sets = [numpy.empty((n_sets, d)) for _ in range(n_per_set)]
    for i in range(n_sets):
        y_i = numpy.random.choice([0, 1])
        y_encoded[i, y_i] = 1
        indices = numpy.random.choice(numpy.arange(n_samples)[y == y_i], size=n_per_set)
        for j in range(n_per_set):
            sets[j][i] = X[indices[j]]

    # Model
    model = model_mk_rff(input_dimensions={d: n_per_set}, embedding_dim=embedding_dim, n_classes=n_classes)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    # Fit & predict
    model.fit(sets, y_encoded, batch_size=128, epochs=50, verbose=True)
    eval_model = model.evaluate(sets, y_encoded, verbose=False)
    print("Correct classification rate:", eval_model[1])

    # Just check that weights are shared, not repeated as many times as the number of features in the sets
    print("Weights:", [w.shape for w in model.get_weights()])

    del model
