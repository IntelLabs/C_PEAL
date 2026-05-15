import numpy as np
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance

def compute_virtual_logit_match(feature_id_train, feature_id_val, logit_id_train, logit_id_val, w, b, DIM):
    """
    Compute the virtual logit match score for a validation set.

    Args:
        feature_id_train (ndarray): Training features, shape (n_train_samples, n_features).
        feature_id_val (ndarray): Validation features, shape (n_val_samples, n_features).
        logit_id_train (ndarray): Logit outputs for the training set, shape (n_train_samples, n_classes).
        logit_id_val (ndarray): Logit outputs for the validation set, shape (n_val_samples, n_classes).
        w (ndarray): Weight matrix, shape (n_features, n_features) or compatible for pinv operation.
        b (ndarray): Bias vector, shape (n_features,).
        DIM (int): Dimension index to select the eigen vectors for low-variance subspace.

    Returns:
        score_id (ndarray): Computed score for the validation set, shape (n_val_samples,).
    """
    
    u = -np.matmul(pinv(w), b)   # Calculate the mean vector u using the pseudoinverse of w and vector b
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)

    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
    
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    # print(f'{alpha=:.4f}')

    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val

    return score_id