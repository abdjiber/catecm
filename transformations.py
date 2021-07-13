import numpy as np
import pandas as pd
import itertools
from helpers import normalize

def findsubsets(s, n):
    return list(itertools.combinations(s, n))


def pignistic_transformation(credal_p, focalsets):
    """Transform a credal partition to a fuzzy partition.

    Parameters
    ----------
    credal_p : ndarray (n_samples, n_focalsets)
            The credal partition.

    focalsets : array of length n_focalsets
        The folcalsets.

    Returns
    -------
    betp : ndarray (n_samples, n_clusters)
            The fuzzy partition obtained from the credal partition.
    """
    n_samples = credal_p.shape[0]
    singletons = [fs for fs in focalsets if len(fs) == 1]
    betp = np.zeros((n_samples, len(singletons)), dtype='float')
    credal_p_norm = normalize(credal_p)
    for i in range(n_samples):
        for j, fs in enumerate(singletons):
            for h, set_ in enumerate(focalsets):
                if fs[0] in set_:
                    betp[i, j] += credal_p_norm[i, h] / float(len(set_))
    betp = pd.DataFrame(betp, columns=singletons)
    return betp


def bel(credal_p, focalsets, type_fs, n_clusters):
    """Compute a belief partition from a credal partition.

    Parameters
    ----------
    credal_p : ndarray (n_samples, n_focalsets)
            The credal partition.

    focalsets : array of length n_focalsets
        The folcalsets.

    type_fs : "singleton", "doublon" or "all", default="doublon"
        Specify the type of desired focal sets. If the value is "singletons", then the focal sets are: the empty set
        , focal sets with size 1 (i.e., clusters) and Omega. If the value is "doublon" the focal sets are: the empty set, focal sets with size less than two
        and Omega. If the value is "all", then all the focal sets are used.

    n_clusters : int
       The number of desired clusters.

    Returns
    -------
    bel_p : ndarray of shape (n_samples, n_focalsets)
        The belief partition obtained from the credal partition.
    """
    init = np.zeros(credal_p.shape)
    bel_p = pd.DataFrame(init, columns=focalsets)
    for i, fs in enumerate(focalsets):
        if len(fs) == 1:
            bel_p.iloc[:, i] += credal_p[:, list(fs)].sum(axis=1)
        else:
            subsets = np.array(
                [findsubsets(fs, n) for n in range(1,
                                                   len(fs) + 1)],
                dtype=object)
            subs = []
            for subset in subsets:
                for s in subset:
                    subs.append(s)
            for s in subs:
                if type_fs == "doublon" and (len(s) <= 2 or len(s) == n_clusters):
                    bel_p.iloc[:, i] += credal_p[:, focalsets.index(s)]
                elif type_fs == "plain":
                    bel_p.iloc[:, i] += credal_p[:, focalsets.index(s)]
    return bel_p


def pl(credal_p, focalsets):
    """Compute the plausibility partition from the credal partition.

    Parameters
    ----------
    credal_p : ndarray (n_samples, n_focalsets)
            The credal partition.

    focalsets : array of length n_focalsets
        The folcalsets.

    Returns
    -------
    pl_p : ndarray of shape (n_samples, n_focalsets)
            The plausibility partition obtained from the credal partition.

    """
    init = np.zeros(credal_p.shape)
    pl_p = pd.DataFrame(init, columns=focalsets)
    for i, fs in enumerate(focalsets[1:]):
        for j, fs2 in enumerate(focalsets):
            if set(fs).intersection(fs2) != set():
                pl_p.iloc[:, i + 1] += credal_p[:, j]
    return pl_p


