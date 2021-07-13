import numpy as np


def nonspecificity(credal_p, focalsets):
    """Compute the nonspecificity of a credal partition.

    Parameters
    ----------
    credal_p : ndarray (n_samples, n_focalsets)
            The credal partition.

    focalsets : array of length n_focalsets
        The folcalsets.

    Returns
    -------
    N : float
        The nonspecificity of the credal partition.

    """
    n_samples = credal_p.shape[0]
    len_fs = [len(fs) for fs in focalsets if fs != tuple()]
    len_fs = np.array(len_fs)
    N = np.sum(len_fs * credal_p[:, 1:]) + np.sum(
        credal_p[:, 0] * np.log2(max(len_fs)))
    N /= n_samples * np.log2(max(len_fs))
    return N
