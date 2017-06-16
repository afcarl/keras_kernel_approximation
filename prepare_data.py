import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def extract_features(X_3d: numpy.ndarray, sz: int, use_time=True):
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
    list of (sz_ts - sz + 1) arrays of shape (n_ts, sz * d) or (n_ts, sz * d + 1) if use_time=True
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


def ecml17_tiselac_data_preparation(X: numpy.ndarray, d=10, feature_sizes=(8, ), use_time=True):
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
        prepared_data.extend(extract_features(X_, sz, use_time=use_time))
    return prepared_data
