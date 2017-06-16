from keras.models import Model
from keras.layers import Dense, Input, Average
from sklearn.datasets import make_circles
import numpy

from layers import CosineActivatedDenseLayer

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def model_mk_rff(input_dim, embedding_dim, n_per_set):
    """Build a match kernel-like model that computes the average of RFF feature for a set and then classifies it using
    Logistic Regression (since the embedding space is supposed to be a good place for linear separability of the sets).

    Parameters
    ----------
    input_dim : int
        Dimension of the input features in the sets
    embedding_dim : int
        Dimension of the RFF output space (approximation of the RKHS associated to the best kernel)
    n_per_set : int
        number of features per set (Fixed here)

    Returns
    -------
    Model
        The full model (including the Logistic Regression step), but not compiled
    """
    inputs = [Input(shape=(input_dim,)) for _ in range(n_per_set)]
    rff_layer = CosineActivatedDenseLayer(units=embedding_dim)
    rffs = [rff_layer(input_feature) for input_feature in inputs]
    avg_rff = Average()(rffs)
    predictions = Dense(units=n_classes, activation="softmax")(avg_rff)

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
    model = model_mk_rff(input_dim=d, embedding_dim=embedding_dim, n_per_set=n_per_set)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # Fit & predict
    model.fit(sets, y_encoded, batch_size=128, epochs=50, verbose=1)
    y_pred = model.predict(sets, verbose=False)
    print(numpy.sum(y_pred.argmax(axis=1) == y_encoded.argmax(axis=1)) / n_sets)

    # Just check that weights are shared, not repeated as many times as the number of features in the sets
    print("Weights:", [w.shape for w in model.get_weights()])

    del model