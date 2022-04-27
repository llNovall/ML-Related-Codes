import numpy as np

## Calculate PCA
def PCA(X, n_components=2):

    X_demeaned = X - np.mean(X, axis = 0)

    covariance_matrix = np.cov(X_demeaned, rowvar=False)

    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)
    
    idx_sorted = np.flip(np.argsort(eigen_vals))

    eigen_vals_sorted = eigen_vals[idx_sorted]
    eigen_vecs_sorted = eigen_vecs[:,idx_sorted]

    eigen_vecs_subset = eigen_vecs_sorted[:,:n_components]

    X_reduced = np.matmul(eigen_vecs_subset.T, X_demeaned.T).T

    return X_reduced


