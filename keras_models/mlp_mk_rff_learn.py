from keras.layers import Dense, Input
from keras.layers.merge import average, concatenate
from keras.models import Model

from utils.layers import RFFLayer

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def model_mlp_mk_rff(input_dimensions, embedding_dim, hidden_classification_layers, n_classes):
    """Build a match kernel-like model that computes the average of RFF feature for a set and then classifies it using
    specified fully connected layers.

    Parameters
    ----------
    input_dimensions : dict
        Dictionary of dimensions of the input features: {dim0: n_features0, dim1: n_features1, ...}
    embedding_dim : int
        Dimension of the RFF output space (approximation of the RKHS associated to the best kernel)
    hidden_classification_layers : list
        Number of units per hidden layer in the fully connected MLP classification part of the model
    n_classes : int
        Number of classes for the classification problem

    Returns
    -------
    Model
        The full model (including the Logistic Regression step), but not compiled

    Examples
    --------
    >>> m = model_mlp_mk_rff(input_dimensions={2: 5}, embedding_dim=10, hidden_classification_layers=[5, 5], n_classes=2)
    >>> m.count_params()
    127
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
    hidden_layers = [concatenated_avg_rffs]
    for n_units in hidden_classification_layers:
        hidden_layers.append(Dense(units=n_units, activation="tanh")(hidden_layers[-1]))
    predictions = Dense(units=n_classes, activation="softmax")(hidden_layers[-1])

    return Model(inputs=inputs, outputs=predictions)
