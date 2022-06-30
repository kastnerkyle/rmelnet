import torch
import numpy as np
def space2batch(x, axis):
    if axis not in [2, 3]:
        raise ValueError("space2batch only operates on axis 2 or 3, as 2D data is assumed (n_batch, channel (depth), Y, X")
    if axis == 2:
        x_flat = x.permute(3, 1, 0, 2).reshape(x.shape[3], x.shape[1], -1)
        x_flat = x_flat.permute(0, 2, 1)
    elif axis == 3:
        x_flat = x.permute(2, 1, 0, 3).reshape(x.shape[2], x.shape[1], -1)
        x_flat = x_flat.permute(0, 2, 1)
    return x_flat.contiguous()


def batch2space(x, n_batch, axis):
    if axis not in [2, 3]:
        raise ValueError("batch2space only operates on axis 2 or 3, as 1D input is assumed (n_batch * n_original_dim, channel (depth), feat")
    x_rec = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], n_batch, x.shape[1] // n_batch)
    if axis == 2:
        x_rec = x_rec.permute(2, 1, 3, 0)
    elif axis == 3:
        x_rec = x_rec.permute(2, 1, 0, 3)
    return x_rec.contiguous()


def split(x, axis):
    assert x.shape[axis] % 2 == 0
    if axis == 0:
        return x[0::2].contiguous(), x[1::2].contiguous()
    elif axis == 1:
        return x[:, 0::2].contiguous(), x[:, 1::2].contiguous()
    elif axis == 2:
        return x[:, :, 0::2].contiguous(), x[:, :, 1::2].contiguous()
    elif axis == 3:
        return x[:, :, :, 0::2].contiguous(), x[:, :, :, 1::2].contiguous()
    else:
        raise ValueError("Only currently support axis 0 1 2 or 3")


def split_np(x, axis):
    assert x.shape[axis] % 2 == 0
    if axis == 0:
        return x[0::2], x[1::2]
    elif axis == 1:
        return x[:, 0::2], x[:, 1::2]
    elif axis == 2:
        return x[:, :, 0::2], x[:, :, 1::2]
    elif axis == 3:
        return x[:, :, :, 0::2], x[:, :, :, 1::2]
    else:
        raise ValueError("Only currently support axis 0 1 2 or 3")


def interleave(x_1, x_2, axis):
    if axis == 0:
        c = torch.empty(2 * x_1.shape[0], x_1.shape[1], x_1.shape[2], x_1.shape[3]).to(x_1.device)
        c[::2] = x_1
        c[1::2] = x_2
        return c
    if axis == 1:
        c = torch.empty(x_1.shape[0], 2 * x_1.shape[1], x_1.shape[2], x_1.shape[3]).to(x_1.device)
        c[:, ::2] = x_1
        c[:, 1::2] = x_2
        return c
    if axis == 2:
        c = torch.empty(x_1.shape[0], x_1.shape[1], 2 * x_1.shape[2], x_1.shape[3]).to(x_1.device)
        c[:, :, ::2, :] = x_1
        c[:, :, 1::2, :] = x_2
        return c
    if axis == 3:
        c = torch.empty(x_1.shape[0], x_1.shape[1], x_1.shape[2], 2 * x_1.shape[3]).to(x_1.device)
        c[:, :, :, ::2] = x_1
        c[:, :, :, 1::2] = x_2
        return c

def interleave_np(x_1, x_2, axis):
    if axis == 1:
        c = np.empty((x_1.shape[0], 2 * x_1.shape[1], x_1.shape[2], x_1.shape[3])).astype(x_1.dtype)
        c[:, ::2] = x_1
        c[:, 1::2] = x_2
        return c
    if axis == 2:
        c = np.empty((x_1.shape[0], x_1.shape[1], 2 * x_1.shape[2], x_1.shape[3])).astype(x_1.dtype)
        c[:, :, ::2, :] = x_1
        c[:, :, 1::2, :] = x_2
        return c
    if axis == 3:
        c = np.empty((x_1.shape[0], x_1.shape[1], x_1.shape[2], 2 * x_1.shape[3])).astype(x_1.dtype)
        c[:, :, :, ::2] = x_1
        c[:, :, :, 1::2] = x_2
        return c


def batch_mean_variance_update(X, old_mean, old_variance, old_sample_count):
    """Calculate an average mean update and a Youngs and Cramer variance update.
    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update
    old_mean : array-like, shape: (n_features,)
    old_variance : array-like, shape: (n_features,)
    old_sample_count : int
    Returns
    -------
    updated_mean : array, shape (n_features,)
    updated_variance : array, shape (n_features,)
    updated_sample_count : int
    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample variance:
        recommendations, The American Statistician, Vol. 37, No. 3, pp. 242-247
    """
    new_sum = X.sum(axis=0)
    new_variance = X.var(axis=0) * X.shape[0]
    old_sum = old_mean * old_sample_count
    n_samples = X.shape[0]
    updated_sample_count = old_sample_count + n_samples
    partial_variance = old_sample_count / (n_samples * updated_sample_count) * (
        n_samples / old_sample_count * old_sum - new_sum) ** 2
    unnormalized_variance = old_variance * old_sample_count + new_variance + \
        partial_variance
    return ((old_sum + new_sum) / updated_sample_count,
            unnormalized_variance / updated_sample_count,
            updated_sample_count)
