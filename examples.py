"""
Example of application of the CatECM algorithm.
"""
import numpy as np

from catecm import CatECM

soybean = np.loadtxt(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data",
    delimiter=",",
    dtype="O")
n_features = soybean.shape[1]
features = [f"A{i}" for i in range(1, n_features + 1)]
true_labels = soybean[:, -1]  # The last column corresponds to objects classes.
soybean = np.delete(soybean, n_features - 1, axis=1)
catecm = CatECM(n_clusters=4, alpha=1.1, verbose=False)
catecm.fit(soybean, features)
print("Scores")
print("Nonspecificity: ", catecm.N)
