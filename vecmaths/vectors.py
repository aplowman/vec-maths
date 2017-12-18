"""Module containing functions related to general vector operations."""
import numpy as np


def cross_prod_mat(axis):
    """
    Find the skew-symmetric matrix which allows representing the
    cross product with matrix multiplication.

    Parameters
    ----------
    ax : ndarray

    """

    if axis.ndim == 1:
        axis = axis[:, None]

    if axis.shape[-2:] != (3, 1):
        raise ValueError('Inner shape of `ax` must be (N, 3, 1)'
                         ' or (3, 1) or (3,).')

    ret_shape = list(axis.shape)
    ret_shape[-1] = 3
    ret_shape = tuple(ret_shape)

    ret = np.zeros(ret_shape)

    ret[..., 0, 1] = -axis[..., 2, 0]
    ret[..., 0, 2] = axis[..., 1, 0]
    ret[..., 1, 0] = axis[..., 2, 0]
    ret[..., 1, 2] = -axis[..., 0, 0]
    ret[..., 2, 0] = -axis[..., 1, 0]
    ret[..., 2, 1] = axis[..., 0, 0]

    return ret


def perpendicular(vec, axis=-1):
    """
    Get 3-vectors perpendicular to a given set of 3-vectors.

    Parameters
    ----------
    vec : ndarray
        Array of vectors.
    axis : int, optional
        The axis along which the 3-vectors are defined.

    Returns
    -------
    perp_vec : ndarray
        Array of the same shape as input array `vec`, where each output vector
        is perpendicular to its corresponding input vector.

    """

    if vec.shape[axis] != 3:
        raise ValueError('Size of dimension `axis` ({}) along `vec` must be 3,'
                         ' but is {}.'.format(axis, vec.shape[axis]))

    # Reshape:
    vec = np.swapaxes(vec, -1, axis)

    # Return array will have the same shape as input
    perp_vec = np.ones_like(vec) * np.nan

    # Find where first component magnitudes are larger than last:
    a_gt_c = np.abs(vec[..., 0]) > np.abs(vec[..., 2])
    a_notgt_c = np.logical_not(a_gt_c)

    # Make bool index arrays
    a_gt_c_0 = np.zeros_like(perp_vec, dtype=bool)
    a_gt_c_0[..., 0] = a_gt_c
    a_gt_c_1 = np.roll(a_gt_c_0, shift=1, axis=-1)
    a_gt_c_2 = np.roll(a_gt_c_1, shift=1, axis=-1)

    a_notgt_c_0 = np.zeros_like(perp_vec, dtype=bool)
    a_notgt_c_0[..., 0] = a_notgt_c
    a_notgt_c_1 = np.roll(a_notgt_c_0, shift=1, axis=-1)
    a_notgt_c_2 = np.roll(a_notgt_c_1, shift=1, axis=-1)

    # Set each component of the output vectors:
    perp_vec[a_gt_c_0] = vec[a_gt_c][..., 2]
    perp_vec[a_gt_c_1] = 0
    perp_vec[a_gt_c_2] = -vec[a_gt_c][..., 0]

    perp_vec[a_notgt_c_0] = 0
    perp_vec[a_notgt_c_1] = vec[a_notgt_c][..., 2]
    perp_vec[a_notgt_c_2] = -vec[a_notgt_c][..., 1]

    # Reshape to original shape:
    perp_vec = np.swapaxes(perp_vec, -1, axis)

    return perp_vec
