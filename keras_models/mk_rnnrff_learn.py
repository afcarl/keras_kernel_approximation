from keras.layers import Dense, Input, SimpleRNN
from keras.layers.merge import average, concatenate
from keras.models import Model

from utils.layers import RFFLayer

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def model_mk_rnnrff(input_dimensions, rnn_dim, rff_dim, n_classes):
    """Description yet to come

    Parameters
    ----------
    input_dimensions : dict
        Dictionary of shapes of the input features: {(sz0, dim0): n_features0, (sz1, dim1): n_features1, ...}
    rnn_dim : int
        Dimension of the RNN output space.
    rff_dim : int
        Dimension of the RFF output space (approximation of the RKHS associated to the best kernel)
    n_classes : int
        Number of classes for the classification problem

    Returns
    -------
    Model
        The full model (including the Logistic Regression step), but not compiled
    """
    inputs = []
    rnn_layer = SimpleRNN(units=rnn_dim, activation="tanh")
    rff_layer = RFFLayer(units=rff_dim)
    for sz, d in sorted(input_dimensions.keys()):
        n_features = input_dimensions[sz, d]
        inputs.extend([Input(shape=(sz, d)) for _ in range(n_features)])
    if len(inputs) == 1:
        rff_layer = RFFLayer(units=rff_dim)
        concatenated_avg_rffs = rff_layer(rnn_layer(inputs[0]))
    elif len(input_dimensions) == 1:
        rffs = []
        for input_feature in inputs:
            rffs.append(rff_layer(rnn_layer(input_feature)))
        concatenated_avg_rffs = average(rffs)
    else:
        avg_rffs = []
        idx0 = 0
        for sz, d in sorted(input_dimensions.keys()):
            n_features = input_dimensions[sz, d]
            rffs = [rff_layer(rnn_layer(input_feature)) for input_feature in inputs[idx0:idx0+n_features]]

            avg_rffs.append(average(rffs))
            idx0 += n_features
        concatenated_avg_rffs = concatenate(avg_rffs)
    predictions = Dense(units=n_classes, activation="softmax")(concatenated_avg_rffs)

    return Model(inputs=inputs, outputs=predictions)
