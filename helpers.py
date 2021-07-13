import warnings
from itertools import chain, combinations

import numpy as np

from exceptions import CatECMWarning


def check_params(catecm, X):
    """Check the correcteness of input parameters.

    Parameters
    ----------
    catecm : CatECM
        A CatECM instance.

    X : ndarray of shape (n_samples, n_features)
        The input intances to be clustered.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        If X contains features with one unique category the feature is dropped.
    """
    n_samples = X.shape[0]
    if catecm.n_clusters < 2:
        raise ValueError("n_clusters should be >= 2.")
    if catecm.n_clusters > n_samples:
        raise ValueError(f"n_clusters should be <= {n_samples}")
    if catecm.beta < 1:
        raise ValueError("beta should be > 1.")
    if catecm.alpha < 0:
        raise ValueError("alpha should be > 0.")
    if catecm.epsillon < 0:
        raise ValueError("epsillon should be > 0.")
    if not isinstance(catecm.verbose, bool):
        raise ValueError("verbose should be a Boolean.")
    attr_with_one_uniq_val = list()
    for l in range(X.shape[1]):
        _, uniq_vals = np.unique(X[:, l], return_counts=True)
        n_l = len(uniq_vals)
        if n_l == 1:
            attr_with_one_uniq_val.append(l)
    if attr_with_one_uniq_val:
        message = f"Attributes {attr_with_one_uniq_val} contain one unique\
            value,they will be dropped before training."

        warnings.warn(message, category=CatECMWarning, stacklevel=0)
    X = np.delete(X, attr_with_one_uniq_val, axis=1)
    return X


def get_focalsets(n_clusters, type_fs):
    """Generate folcal sets.

    Parameters
    ----------
    n_clusters : int
        The number of desired clusters.

    type_fs : "singleton", "doublon" or "all", default="doublon"
        Specify the type of desired focal sets. If the value is "singletons",
        then the focal sets are: the empty set, focal sets with size 1
        (i.e., clusters) and Omega. If the value is "doublon" the focal sets are:
        the empty set, focal sets with size less than two and Omega.
        If the value is "all", then all the focal sets are used.

    Returns
    -------
    foclasets : array of length n_clusters + 2 if type_fs="singleton",
        (n_cluster! /(n_clusters - 2)!*2!) + 2 and 2^n_clusters if type_fs="all".
        The generated focal sets.
    """
    omega = list(range(1, n_clusters + 1))
    focalsets = list(
        chain.from_iterable(
            combinations(omega, i) for i in range(n_clusters + 1)))
    if type_fs == 'doublon':
        focalsets = [
            fs for fs in focalsets if len(fs) <= 2 or len(fs) == n_clusters
        ]
    elif type_fs == 'singleton':
        focalsets = [
            fs for fs in focalsets if len(fs) <= 1 or len(fs) == n_clusters
        ]
    return focalsets


def get_randn(n_l):
    """Generate random values.

    Parameters
    ----------
    n_l : int
        The length of random numbers to generate.

    Returns
    -------
    randn : array of shape n_l.
        The generated list of random numbers.
    """
    randn = np.abs(np.random.randn(n_l))
    randn /= np.sum(randn)
    return randn


def get_dom_vals_and_size(X):
    """Get the feature domains and size.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training instances to cluster.

    Returns
    -------
    dom_vals : array of shape n_unique_vals
        The domains of the features.

    n_attr_doms : int
        The length of the number of categories of X.
    """
    dom_vals = []
    n_attr_doms = []
    n_features = X.shape[1]
    for k in range(n_features):
        unique = list(np.unique(X[:, k]))
        dom_vals += unique
        n_attr_doms += [len(unique)]
    return dom_vals, n_attr_doms


def normalize(credal_p, type_norm="Dempster"):
    """Perfom a normalization of a credal partition.

    Parameters
    ----------
    credal_p : ndarray of shape (n_samples, n_focalsets)
        The credal partition.

    type_norm : "Dempster" or "Yager", default="Dempster"
        The type of normalization to perform. Possible values are "Dempster"
        and "Yager".

    Returns
    -------
    credal_p_norm : ndarray of shape (n_samples, n_focalsets)
        The normed credal partition.
    """
    credal_p_norm = credal_p.copy()
    if type_norm not in ("Dempster", "Yager"):
        raise ValueError(
            "The parameter type_norm should be either Dempster or Yager")
    if type_norm == "Dempster":
        credal_p_norm[:, 1:] /= np.array([1 - credal_p_norm[:, 0]]).T
    else:
        credal_p_norm[:, -1] += credal_p_norm[:, 0]
        credal_p_norm[:, 0] = 0
    return credal_p_norm
