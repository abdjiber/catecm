import os
import sys

import numpy as np
import pandas as pd

import helpers
import metrics
import transformations
from exceptions import ModelNotFittedError


class CatECM:
    """Categorical Evidential c-emans algorithm.

    Parameters
    ----------
    n_clusters : int
        The number of desired clusters.

    alpha : float
        Weighting exponent to penalize focal sets with high elements. The value of alpha should be > 1.

    beta : float, default=1.2
        The fuzziness weigthing exponent. The default value is 1.2.

    delta : float, default=10
        The distance to the empty set i.e. if the distance between an object and a cluster is greater than delta, the object is considered as an outlier.

    type_fs : "singleton", "doublon" or "all", default="doublon"
        Specify the type of desired focal sets. If the value is "singletons", then the focal sets are: the empty set
        , focal sets with size 1 (i.e., clusters) and Omega. If the value is "doublon" the focal sets are: the empty set, focal sets with size less than two
        and Omega. If the value is "all", then all the focal sets are used.

    epsillon : float, default=1e-3
        The stop criteria i.e., if the absolute difference between two consecutive inertia is less than epsillon, then the algorithm will stop.

    seed : int, default=None.
        The random state seed. This parameter can be set for a replication of the results.

    verbose : Boolean, default=False
        Verbosity mode.

    Attributes
    ----------
    focalsets : array of length n_clusters + 2 if type_fs="singleton", (n_cluster! /(n_clusters - 2)!*2!) + 2 and 2^n_clusters if type_fs="all".
        The generated focal sets.

    credal_partition : ndarray of shape (n_samples, n_focalsets)
        The obtained credal (i.e., evidential partition).

    N : float
        The nonspecificity of the credal partition.

    centers : ndarray of shape (n_attr_doms, n_clusters)
        The obtained cluster centers.

    n_iter : int
        The number of itertations at the end of the algorithm.

    history : array of length n_iter
        The history of the _costs (i.e., inertia) values.

    betp : ndarray of shape (n_samples, _clusters)
        The belief to probability matrix obtained from a pignistic transformation.

    crisp_labels : array of length n_samples
        The crisp labels obtained with a maximum rule from the betp matrix.

    pl : ndarray of shape (n_samples, n_folcalsets)
        The plausibility matrix.

    bel : ndarray of shape (n_samples, n_focalsets)
        The belief matrix.

    Returns
    -------
    A CatECM instance.
    """
    def __init__(self,
                 n_clusters,
                 alpha,
                 beta=1.2,
                 delta=10,
                 type_fs="doublon",
                 epsillon=1e-3,
                 seed=None,
                 verbose=False):
        self.n_clusters = n_clusters
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.type_fs = type_fs
        self.epsillon = epsillon
        self.seed = seed
        self.verbose = verbose
        self.focalsets = helpers.get_focalsets(self.n_clusters, self.type_fs)
        self.n_focalsets = len(self.focalsets)
        self._is_fitted = False

    def _init_centers_singletons(self):
        """Initialize the centers of clusters."""
        if self.seed:
            np.random.seed(self.seed)
        w0 = np.zeros((self.n_attr_doms, self.n_focalsets), dtype='float')
        for j in range(1, self.n_clusters + 1):
            k = 0
            l = 0
            for n_l in self.size_attr_doms:
                l += n_l
                rand_num = helpers.get_randn(n_l)
                w0[k:l, j] = rand_num
                k = l
        return w0

    def _update_centers_focalsets_gt_2(self, w):
        """Update the centers of focal sets with size greater than two."""
        for i in range(self.n_clusters + 1, self.n_focalsets):
            idx = list(self.focalsets[i])
            w[:, i] = w[:, idx].mean(axis=1)
        return w

    def _distance_objects_to_centers(self, X, w):
        """Compute the distance between objects and clusters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.
        w : ndarray of shape (n_attr_doms, n_clusters)
            The centers of clusters.

        Returns
        -------
        dist : np.array
            The distances between objects and clusters.
        """
        dim_dist = self.n_focalsets - 1
        dist = np.zeros((self.n_samples, dim_dist), dtype='float')
        for i in range(self.n_samples):
            xi = X[i]
            for j in range(dim_dist):
                sum_ = 0.0
                k = 0
                l = 0
                for x_l, n_l in zip(xi, self.size_attr_doms):
                    l += n_l
                    dom_val = np.array(self._dom_vals[k:l])
                    w_ = np.array(w[k:l, j])
                    sum_ += 1 - np.sum(w_[dom_val == x_l])
                    k += n_l
                dist[i, j] = sum_ / len(self.focalsets[j + 1])
        return dist

    def _get_credal_partition(self, dist):
        """Compute the credal partition from the distances between objects and cluster centers."""
        power_alpha = -self.alpha / (self.beta - 1)
        power_beta = -2.0 / (self.beta - 1)
        dim_m = self.n_focalsets
        credal_p = np.zeros((self.n_samples, dim_m), dtype='float')
        for i in range(self.n_samples):
            if 0 in dist[i, :]:
                credal_p[i, 1:] = 0
                idx_0 = dist[i, :].tolist().index(0)
                #  If the index in dist is i, the index in m is i + 1 as dim(m) = dim(dist) + 1
                idx_0 += 1
                credal_p[i, idx_0] = 1
            else:
                sum_dij = np.sum([
                    len(self.focalsets[k + 1])**power_alpha *
                    dist[i, k]**power_beta for k in range(dim_m - 1)
                ])
                for j in range(1, dim_m):
                    len_fs = len(self.focalsets[j])
                    credal_p[i, j] = (len_fs**power_alpha *
                                      dist[i, j - 1]**power_beta) / (
                                          sum_dij + self.delta**power_beta)
        credal_p[:, 0] = 1 - np.sum(credal_p[:, 1:], axis=1)
        credal_p = np.where(credal_p < np.finfo("float").eps, 0, credal_p)
        return credal_p

    def _update_cluster_center(self, X, mbeta, w_jl, j, l, a_l, n_l):
        """Update the center for the jth cluster restricted to the lth feature.

        Paramaters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.

        mbeta : ndarray of shape (n_samples, n_focalsets)
            The partition matrix power m.

        w_jl : ndarray
            The jth center restricted to the feature l.

        j : int
            The jth center to be updated.

        l : int
            The lth feature of the jth center to be updated.

        a_l : ndarray of shape (1, n_uniq_val)
            The domain of the lth feature.

        n_l : int
            The length of a_l.

        Returns
        -------
        w_jl : ndarray
            The updated jth center restricted to the lth feature.
        """
        attr_values_freq = np.zeros((n_l), dtype="float")
        for t in range(n_l):
            len_fs = len(self.focalsets[j])
            freq = np.sum(mbeta[np.array(X[:, l]) == a_l[t], j])
            attr_values_freq[t] = len_fs**(self.alpha - 1) * freq
        idx_max_freq = np.argmax(attr_values_freq)
        w_jl[idx_max_freq] = 1
        return w_jl

    def _update_centers_singletons(self, x, credal_p):
        """Update the centers of singletons.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.

        credal_p : ndarray (n_samples, n_focalsets)
            The credal partition.

        Returns
        -------
        w : ndarray of shape (n_attr_doms, n_folcalsets)
            The updated centers of singletons.
        """
        try:
            mbeta = credal_p**self.beta
            w = np.zeros((self.n_attr_doms, self.n_focalsets), dtype='float')
            for j in range(1, self.n_clusters + 1):
                s = 0
                z = 0
                for l, n_l in enumerate(self.size_attr_doms):
                    s += n_l
                    w_jl = w[z:s, j]
                    a_l = self._dom_vals[z:s]
                    w[z:s,
                      j] = self._update_cluster_center(x, mbeta, w_jl, j, l,
                                                       a_l, n_l)
                    z = s
        except RuntimeWarning:
            exit()
        return w

    def _cost(self, dist, credal_p):
        """Compute the cost (intertia) from an iteration.

        Parameters
        ----------
        dist : ndarray of shape (n_samples, n_clusters)
            The distance between objects and clusters.

        credal_p : ndarray of shape (n_samples, n_focalsets)
            The credal partition matrix.

        Returns
        -------
        cost : float
            The cost of the current iteration.
        """
        len_fs = np.array([len(fs) for fs in self.focalsets[1:]])
        bba = np.copy(credal_p)
        bba_power = np.where(bba > 0, bba**self.beta, bba)
        cost = np.sum(len_fs**self.alpha * bba_power[:, 1:] *
                      dist**2.) + np.sum(self.delta**2. * bba_power[:, 0])
        return cost

    def fit(self, X, features):
        """Fit the CatECM model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.

        features : array of shape n_features
            The features names.

        Returns
        -------
        self
            Fitted model.
        """
        self.attr_names = features
        X = helpers.check_params(self, X)
        self.n_samples, n_features = X.shape
        self._dom_vals, self.size_attr_doms = helpers.get_dom_vals_and_size(X)
        self.n_attr_doms = np.sum(self.size_attr_doms)
        w0 = self._init_centers_singletons()
        w = self._update_centers_focalsets_gt_2(w0)
        old_inertia = np.inf
        is_finished = True
        n_iter = 0
        history = []
        while is_finished:
            dist = self._distance_objects_to_centers(X, w[:, 1:])
            credal_p = self._get_credal_partition(dist)
            w = self._update_centers_singletons(X, credal_p)
            w = self._update_centers_focalsets_gt_2(w)
            new_inertia = self._cost(dist, credal_p.copy())
            history.append(new_inertia)
            is_finished = np.abs(old_inertia - new_inertia) > self.epsillon
            old_inertia = new_inertia
            n_iter += 1
            if new_inertia > old_inertia:
                break
        self._is_fitted = True
        self.credal_partition = pd.DataFrame(credal_p, columns=self.focalsets)
        self.centers = pd.DataFrame(w, columns=self.focalsets)
        self.inertia = new_inertia
        self.n_iter = n_iter
        self.betp = transformations.pignistic_transformation(credal_p.copy(), self.focalsets)
        self.crisp_labels = np.argmax(self.betp.values, axis=1)
        self.bel = transformations.bel(credal_p, self.focalsets, self.type_fs,
                                       self.n_clusters)
        self.pl = transformations.pl(credal_p, self.focalsets)
        self.history = history
        self.N = metrics.nonspecificity(credal_p, self.focalsets)

    def predict(self, X):
        """Perfom a prediction new objects intances.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features).
            The new objects instances to be clustered.

        Returns
        -------
        credal_p: array of shape (n_samples, n_focalsets)
            The predicted credal partition.

        w : w : ndarray of shape (n_attr_doms, n_folcalsets)
            The updated centers of clusters.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Please fit the model before using the predict method.")
        self.n_samples = X.shape[0]
        dist = self._distance_objects_to_centers(X, self.centers.values)
        credal_p = self._get_credal_partition(dist)
        w = self._update_centers_singletons(X, credal_p)
        w = self._update_centers_focalsets_gt_2(w)
        return credal_p, w
