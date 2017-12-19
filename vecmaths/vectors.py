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


def vecpair_angle(veca, vecb, ref=None, axis=-1, degrees=False):
    """Find the signed angles between a set of vectors and another.

    Parameters
    ----------
    veca : ndarray
        Inner shape (3,)
    vecb : ndarray
        Inner shape (3,)
    axis : int, optional
        The axis along which each 3-vector is defined. By default, the last
        axis.
    ref : ndarray of size 3 or None, optional
        Reference vector used to determine which +/- axis to use for the
        rotation. If ndarray, must have size 3. For each vector pair, the
        rotation axis closest to the reference vector will be chosen. If None,
        the reference vector is taken as the cross product of each vector pair,
        i.e. all returned angles will be positive. By default, None.
    degrees : bool, optional
        If True returned angles are in degrees, else in radians. By default,
        False.

    Returns
    -------
    theta : ndarray

    """

    # Reshape
    veca = np.swapaxes(veca, -1, axis)
    vecb = np.swapaxes(vecb, -1, axis)

    vecx = np.cross(veca, vecb, axis=-1)
    vecx_normd = vecx / np.linalg.norm(vecx, axis=-1)[..., None]

    if ref is not None:
        ref = np.squeeze(ref)
        ref_normd = ref / np.linalg.norm(ref, axis=-1)
        ref_dot = np.einsum('i,...i->...', ref_normd, vecx_normd)
        vecx_normd[ref_dot < 0] *= -1

    dot_spec = '...i,...i->...'
    dot = np.einsum(dot_spec, veca, vecb)
    tri_prod = np.einsum(dot_spec, vecx, vecx_normd)
    theta = np.arctan2(tri_prod, dot)

    if degrees:
        theta = np.rad2deg(theta)

    return theta
