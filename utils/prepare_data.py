import numpy
from keras.utils import to_categorical

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def extract_features(X_3d, sz, use_time=True, make_monodim=True):
    """
    Parameters
    ----------
    X_3d : array-like of shape (n_ts, sz_ts, d)
        Input data.
    sz : int
        Size (number of time steps) of feature window.
    use_time : bool, default: True
        Whether a last dimension corresponding to time should be appended to the features

    Returns
    -------
    list of (sz_ts - sz + 1) arrays of shape (n_ts, sz * d) if use_time=False or (n_ts, sz * d + 1) if use_time=True
        List of transformed arrays. For each time step, if use_time=True, time information is appended at the end of
        the feature vector.

    Examples
    --------
    >>> X = numpy.arange(10 * 15 * 3).reshape((10, 15, 3))
    >>> [f.shape for f in extract_features(X, sz=14, use_time=False)]
    [(10, 42), (10, 42)]
    >>> [f.shape for f in extract_features(X, sz=14, use_time=True)]
    [(10, 43), (10, 43)]
    """
    if not make_monodim:
        return extract_features_multidim(X_3d, sz)
    n_ts, sz_ts, d = X_3d.shape
    if use_time:
        shape_ret = (n_ts, sz * d + 1)
    else:
        shape_ret = (n_ts, sz * d)
    X_ret = [numpy.zeros(shape_ret) for _ in range(sz_ts - sz + 1)]
    for t in range(sz_ts - sz + 1):
        X_ret[t][:, :sz*d] = X_3d[:, t:t+sz, :].reshape((n_ts, -1))
        if use_time:
            X_ret[t][:, -1] = t / (sz_ts - sz + 1)
    return X_ret


def extract_features_multidim(X_3d, sz):
    """
    Parameters
    ----------
    X_3d : array-like of shape (n_ts, sz_ts, d)
        Input data.
    sz : int
        Size (number of time steps) of feature window.
    use_time : bool, default: True
        Whether a last dimension corresponding to time should be appended to the features

    Returns
    -------
    list of (sz_ts - sz + 1) arrays of shape (n_ts, sz * d) if use_time=False or (n_ts, sz * d + 1) if use_time=True
        List of transformed arrays. For each time step, if use_time=True, time information is appended at the end of
        the feature vector.
    """
    n_ts, sz_ts, d = X_3d.shape
    shape_ret = (n_ts, sz, d)
    X_ret = [numpy.zeros(shape_ret) for _ in range(sz_ts - sz + 1)]
    for t in range(sz_ts - sz + 1):
        X_ret[t] = X_3d[:, t:t+sz, :]
    return X_ret


def ecml17_tiselac_data_preparation(X, d=10, feature_sizes=(8, ), use_time=True, make_monodim=True):
    """
    Examples
    --------
    >>> X = numpy.empty((82230, 230))
    >>> feats_8 = ecml17_tiselac_data_preparation(X, d=10, feature_sizes=(8, ), use_time=False)
    >>> len(feats_8)
    16
    >>> feats_8[0].shape
    (82230, 80)
    >>> feats_8_12 = ecml17_tiselac_data_preparation(X, d=10, feature_sizes=(8, 12), use_time=True)
    >>> len(feats_8_12)
    28
    >>> feats_8_12[0].shape
    (82230, 81)
    >>> feats_8_12[16].shape
    (82230, 121)
    """
    X_ = X.reshape((X.shape[0], -1, d))
    prepared_data = []
    for sz in list(feature_sizes):
        prepared_data.extend(extract_features(X_, sz, use_time=use_time, make_monodim=make_monodim))
    return prepared_data


def load_tiselac(training_set=True, shuffle=False, random_state=None):
    if training_set:
        X = numpy.loadtxt("data_tiselac/training.txt", dtype=numpy.float, delimiter=",")
        X /= X.max()
        X_coord = numpy.loadtxt("data_tiselac/coord_training.txt", dtype=numpy.float, delimiter=",")
        X_coord /= X_coord.max(axis=0)
        y = numpy.loadtxt("data_tiselac/training_class.txt", delimiter=",").astype(numpy.int) - 1
        y = to_categorical(y)
        if shuffle:
            X, X_coord, y = shuffle_data(X, X_coord, y, random_state=random_state)
        return X, X_coord, y
    else:
        X = numpy.loadtxt("data_tiselac/test.txt", dtype=numpy.float, delimiter=",")
        X /= X.max()
        X_coord = numpy.loadtxt("data_tiselac/coord_test.txt", dtype=numpy.float, delimiter=",")
        X_coord /= X_coord.max(axis=0)
        return shuffle_data(X, X_coord)


def shuffle_data(*args, **kwargs):
    rs = kwargs.get("random_state", None)
    if not isinstance(rs, numpy.random.RandomState):
        rs = numpy.random.RandomState(rs)
    assert len(args) > 0
    for X in args[1:]:
        assert X.shape[0] == args[0].shape[0]
    indices = rs.permutation(args[0].shape[0])
    return [X[indices] for X in args]
