# About
This repository contains the Python implementation of my research paper algorithm **categorical evidential c-means** (cat-ECM).

# Examples
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

# Citation
If you use this work please cite the following paper.
> A. J. Djiberou Mahamadou, V. Antoine, G. J. Christie and S. Moreno, "Evidential clustering for categorical data," 2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), New Orleans, LA, USA.
