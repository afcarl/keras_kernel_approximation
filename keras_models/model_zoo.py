from keras.layers import Dense, Input, SimpleRNN, LSTM, concatenate
from keras.models import Model

from utils.layers import RFFLayer

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def model_mlp(input_shape, hidden_layers, n_classes=None, activation="relu"):
    mlp_input = Input(shape=input_shape, name="input_mlp")
    output_layer = mlp_input
    for i, n_units in enumerate(hidden_layers):
        output_layer = Dense(units=n_units, activation=activation, name="hidden_mlp_%d" % i)(output_layer)
    if n_classes is not None:
        preds = Dense(units=n_classes, activation="softmax", name="softmax_mlp")(output_layer)
    else:
        preds = output_layer
    return Model(inputs=mlp_input, outputs=preds)


def model_mlp_rff(input_shape, hidden_layers, rff_layer_dim, n_classes=None, activation="relu"):
    mlp_input = Input(shape=input_shape, name="input_mlp")
    output_layer = mlp_input
    for i, n_units in enumerate(hidden_layers):
        output_layer = Dense(units=n_units, activation=activation, name="hidden_mlp_%d" % i)(output_layer)
    rff_layer = RFFLayer(units=rff_layer_dim, activation=activation, name="rff_mlp")(output_layer)
    if n_classes is not None:
        preds = Dense(units=n_classes, activation="softmax", name="softmax_mlp")(rff_layer)
    else:
        preds = rff_layer
    return Model(inputs=mlp_input, outputs=preds)


def model_rnn(input_shape, hidden_layers, rnn_layer_dim, input_shape_side_info=None, n_classes=None, activation="relu",
              use_lstm=False):
    rnn_input = Input(shape=input_shape, name="input_rnn")
    inputs = [rnn_input]
    if use_lstm:
        rnn_layer = LSTM(units=rnn_layer_dim, activation=activation, name="lstm")(rnn_input)
    else:
        rnn_layer = SimpleRNN(units=rnn_layer_dim, activation=activation, name="rnn")(rnn_input)
    if input_shape_side_info is not None:
        rnn_input_side_info = Input(shape=input_shape_side_info, name="input_rnn_side_info")
        inputs.append(rnn_input_side_info)
        rnn_output_layer = concatenate([rnn_layer, rnn_input_side_info], name="rnn_side_info_concatenated")
    else:
        rnn_output_layer = rnn_layer
    for i, n_units in enumerate(hidden_layers):
        rnn_output_layer = Dense(units=n_units, activation=activation, name="hidden_rnn_%d" % i)(rnn_output_layer)
    if n_classes is not None:
        preds = Dense(units=n_classes, activation="softmax", name="softmax_rnn")(rnn_output_layer)
    else:
        preds = rnn_output_layer
    return Model(inputs=inputs, outputs=preds)