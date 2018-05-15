"""Module containing functions related to general vector operations.

TODO: use `normalise` throughout this module.

"""
from itertools import permutations, combinations
import numpy as np


def normalise(vecs, axis=-1):
    """Normalise an n-dimensional array of vectors.

    Only non-zero vectors are normalised.

    Parameters
    ----------
    vecs : ndarray of innermost shape (3,)
        The innermost axis defines the three vector.
    axis : int, optional
        The axis which defines the vectors. Set to the last 
        axis (-1) by default.

    Returns
    -------
    vecs_normd : ndarray with same shape as input array

    """
    if axis < 0:
        axis = len(vecs.shape) + axis

    norm = np.linalg.norm(vecs, axis=axis)

    vecs_shp_pre_ax = vecs.shape[:axis]
    vecs_shp_post_ax = vecs.shape[axis + 1:]
    norm_shpe = vecs_shp_pre_ax + (1,) + vecs_shp_post_ax

    norm_rs = norm.reshape(norm_shpe)
    norm_rs_rep = np.repeat(norm_rs, vecs.shape[axis], axis=axis)
    norm_non_zero = np.logical_not(np.isclose(norm_rs_rep, 0))

    vecs_normd = np.copy(vecs)
    vecs_normd[norm_non_zero] = (
        vecs_normd[norm_non_zero] / norm_rs_rep[norm_non_zero]
    )

    return vecs_normd


def validate_vecpairs(veca, vecb, axis):
    """Validate the input form of vector pair functions."""

    if veca.shape != vecb.shape:
        raise ValueError('`veca` and `vecb` must have the same shape, but have'
                         ' shapes: {} and {}.'.format(veca.shape, vecb.shape))

    if axis >= veca.ndim:
        raise ValueError('`veca` and `vecb` have {} dimensions, but axis '
                         'is: {}.'.format(veca.ndim, axis))

    if veca.shape[axis] != 3:
        raise ValueError('The `axis` ({}) dimension of `veca` and `vecb` must '
                         'define the axis along which the three-vectors are '
                         'defined and must have length 3, but have '
                         'length {}.'.format(axis, veca.shape[axis]))


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
        The first set of vectors.
    vecb : ndarray
        The second set of vectors.
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

    validate_vecpairs(veca, vecb, axis)

    # Reshape
    veca = np.swapaxes(veca, -1, axis)
    vecb = np.swapaxes(vecb, -1, axis)

    vecx = np.cross(veca, vecb, axis=-1)
    vecx_normd = normalise(vecx)

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


def vecpair_cos(veca, vecb, axis=-1):
    """Find the cosines between a set of vectors and another.

    Parameters
    ----------
    veca : ndarray
        The first set of vectors.
    vecb : ndarray
        The second set of vectors.
    axis : int, optional
        The axis along which each 3-vector is defined. By default, the last
        axis.

    Returns
    -------
    cosines : ndarray
        Array with the same shape as input vectors except for the removed
        `axis` dimension.

    """
    validate_vecpairs(veca, vecb, axis)

    # Reshape
    veca = np.swapaxes(veca, -1, axis)
    vecb = np.swapaxes(vecb, -1, axis)

    veca_normd = veca / np.linalg.norm(veca, axis=-1)[..., None]
    vecb_normd = vecb / np.linalg.norm(vecb, axis=-1)[..., None]

    cosines = np.einsum('...i,...i->...', veca_normd, vecb_normd)[..., None]

    # Reshape to original axis order
    cosines = np.swapaxes(cosines, -1, axis).squeeze(axis=axis)

    return cosines


def vecpair_sin(veca, vecb, axis=-1):
    """Find the sines between a set of vectors and another.

    Parameters
    ----------
    veca : ndarray
        The first set of vectors.
    vecb : ndarray
        The second set of vectors.
    axis : int, optional
        The axis along which each 3-vector is defined. By default, the last
        axis.

    Returns
    -------
    sines : ndarray
        Array with the same shape as input vectors except for the removed
        `axis` dimension.

    """

    validate_vecpairs(veca, vecb, axis)

    # Reshape
    veca = np.swapaxes(veca, -1, axis)
    vecb = np.swapaxes(vecb, -1, axis)

    veca_normd = veca / np.linalg.norm(veca, axis=-1)[..., None]
    vecb_normd = vecb / np.linalg.norm(vecb, axis=-1)[..., None]

    xprod = np.cross(veca_normd, vecb_normd, axis=-1)

    # Reshape to original axis order
    xprod = np.swapaxes(xprod, -1, axis)
    sines = np.linalg.norm(xprod, axis=axis)

    return sines


def find_positive_int_vecs(search_size, dim=3):
    """
    Find arbitrary-dimension positive integer vectors which are
    non-collinear whose components are less than or equal to a
    given search size. Vectors with zero components are not included.

    Non-collinear here means no two vectors are related by a scaling factor.

    Parameters
    ----------
    search_size : int
        Positive integer which is the maximum vector component
    dim : int
        Dimension of vectors to search.

    Returns
    -------
    ndarray of shape (N, `dim`)

    """

    # Generate trial vectors as a grid of integer vectors
    search_ints = np.arange(1, search_size + 1)
    search_grid = np.meshgrid(*(search_ints,) * dim)
    trials = np.vstack(search_grid).reshape((dim, -1)).T

    # Multiply each trial vector by each possible integer up to
    # `search_size`:
    search_ints_rs = search_ints.reshape(-1, 1, 1)
    trial_combs = trials * search_ints_rs

    # Combine trial vectors and their associated scaled vectors:
    trial_combs_all = np.vstack(trial_combs)

    # Find unique vectors. The inverse indices`uinv` indexes
    # the set of unique vectors`u` to generate the original array `pv`:
    uniq, uniq_inv = np.unique(trial_combs_all, axis=0, return_inverse=True)

    # For a given set of (anti-)parallel vectors, we want the smallest, so get
    # their relative magnitudes. This is necessary since `np.unique` does not
    # return vectors sorted in a sensible way if there are negative components.
    # (But we do we have negative components here?)
    uniq_mag = np.sum(uniq**2, axis=1)

    # Get the magnitudes of just the directionally-unique vectors:
    uniq_inv_mag = uniq_mag[uniq_inv]

    # Reshape the magnitudes to allow sorting for a given scale factor:
    uniq_inv_mag_rs = np.reshape(uniq_inv_mag, (search_size, -1))

    # Get the indices which sort the trial vectors
    mag_srt_idx = np.argsort(uniq_inv_mag_rs, axis=0)

    # Reshape the inverse indices
    uniq_inv_rs = np.reshape(uniq_inv, (search_size, -1))

    # Sort the inverse indices by their corresponding vector magnitudes,
    # for each scale factor:
    col_idx = np.tile(np.arange(uniq_inv_rs.shape[1]), (search_size, 1))
    uniq_inv_rs_srt = uniq_inv_rs[mag_srt_idx, col_idx]

    # Only keep inverse indices in first row which are not in any other row.
    # First row indexes lowest magnitude vectors for each scale factor.
    idx = np.setdiff1d(uniq_inv_rs_srt[0], uniq_inv_rs_srt[1:])

    # Sort kept vectors by magnitude
    final_mags = uniq_mag[idx]
    final_mags_idx = np.argsort(final_mags)

    ret = uniq[idx][final_mags_idx]

    return ret


def tile_int_vecs(int_vecs, dim):
    """
    Tile arbitrary-dimension integer vectors such that they occupy a
    half-space.

    """
    # For tiling, there will a total of 2^(`dim` - 1) permutations of the
    # original vector set. (`dim` - 1) since we want to fill a half space.
    i = np.ones(dim - 1, dtype=int)
    t = np.triu(i, k=1) + -1 * np.tril(i)

    # Get permutation of +/- 1 factors to tile initial vectors into half-space
    perms_partial_all = [j for i in t for j in list(permutations(i))]
    perms_partial = np.array(list(set(perms_partial_all)))

    perms_first_col = np.ones((2**(dim - 1) - 1, 1), dtype=int)
    perms_first_row = np.ones((1, dim), dtype=int)
    perms_non_eye = np.hstack([perms_first_col, perms_partial])
    perms = np.vstack([perms_first_row, perms_non_eye])

    perms_rs = perms[:, np.newaxis]
    tiled = int_vecs * perms_rs
    ret = np.vstack(tiled)

    return ret


def find_non_parallel_int_vecs(search_size, dim=3, tile=False):
    """
    Find arbitrary-dimension integer vectors which are non-collinear, whose
    components are less than or equal to a given search size.

    Non-collinear here means no two vectors are related by a scaling factor.
    The zero vector is excluded.

    Parameters
    ----------
    search_size : int
        Positive integer which is the maximum vector component.
    dim : int
        Dimension of vectors to search.
    tile : bool, optional
        If True, the half-space of dimension `dim` is filled with vectors,
        otherwise just the positive vector components are considered. The
        resulting vector set will still contain only non-collinear vectors.

    Returns
    -------
    ndarray of shape (N, `dim`)
        Vectors are not globally ordered.

    Notes
    -----
    Searching for vectors with `search_size` of 100 uses about 9 GB of memory.

    """

    # Find all non-parallel positive integer vectors which have no
    # zero components:
    ret = find_positive_int_vecs(search_size, dim)

    # If requested, tile the vectors such that they occupy a half-space:
    if tile and dim > 1:
        ret = tile_int_vecs(ret, dim)

    # Add in the vectors which are contained within a subspace of dimension
    # (`dim` - 1) on the principle axes. I.e. vectors with zero components:
    if dim > 1:

        # Recurse through each (`dim` - 1) dimension subspace:
        low_dim = dim - 1
        vecs_lower = find_non_parallel_int_vecs(search_size, low_dim, tile)

        # Raise vectors to current dimension with a zero component. The first
        # (`dim` - 1) "prinicple" vectors (of the form [1, 0, ...]) should be
        # considered separately, else they will be repeated.
        principle = np.eye(dim, dtype=int)
        non_prcp = vecs_lower[low_dim:]

        if non_prcp.size:

            edges_shape = (dim, non_prcp.shape[0], non_prcp.shape[1] + 1)
            vecs_edges = np.zeros(edges_shape, dtype=int)
            edges_idx = list(combinations(list(range(dim)), low_dim))

            for i in range(dim):
                vecs_edges[i][:, edges_idx[i]] = non_prcp

            vecs_edges = np.vstack([principle, *vecs_edges])

        else:
            vecs_edges = principle

        ret = np.vstack([vecs_edges, ret])

    return ret


def get_parallel_idx(vecs_a, vecs_b):
    """
    Find which vectors in an array of vectors are (anti-)parallel to which
    vectors in another array of vectors, according to their cross product.

    Parameters
    ----------
    vecs_a : ndarray of shape (3, N)
        Array of column vectors
    vecs_b : ndarray of shape (3, M)
        Array of column vectors

    Returns
    -------
    dict of (int: ndarray of int of shape (P,))
        Returned dict has keys which index `vecs_a` and values which are
        integer arrays which index `vecs_b`.

    """

    ret = {}
    for vec_a_idx, vec_a in enumerate(vecs_a.T):

        vec_a = vec_a[:, np.newaxis]
        vec_a_cross = np.cross(vec_a, vecs_b, axis=0)

        is_parallel = np.all(np.isclose(vec_a_cross, 0.0), axis=0)
        parallel_idx = np.where(is_parallel)[0]

        if parallel_idx.size:
            ret.update({
                vec_a_idx: parallel_idx
            })

    return ret


def get_equal_indices(arr, scale_factors=None):
    """
    Return the indices along the first dimension of an array which index equal
    sub-arrays.

    Parameters
    ----------
    arr : ndarray or list
        Array or list of any shape whose elements along its first dimension are
        compared for equality.
    scale_factors : list of float or list of int, optional
        Multiplicative factors to use when checking for equality between
        subarrays. Each factor is checked independently.

    Returns
    -------
    tuple of dict of int: list of int
        Each tuple item corresponds to a scale factor for which each dict maps
        a subarray index to a list of equivalent subarray indices given that
        scale factor. Length of returned tuple is equal to length of
        `scale_factors` or 1 if `scale_factors` is not specified.

    Notes
    -----
    If we have a scale factor `s` which returns {a: [b, c, ...]}, then the
    inverse scale factor `1/s` will return {b: [a], c: [a], ...}.


    Examples
    --------

    1D examples:

    >>> a = np.array([5, 1, 4, 6, 1, 8, 2, 7, 4, 7])
    >>> get_equal_indices(a)
    ({1: [4], 2: [8], 7: [9]},)

    >>> a = np.array([1, -1, -1, 2])
    >>> get_equal_indices(a, scale_factors=[1, -1, -2, -0.5])
    ({1: [2]}, {0: [1, 2]}, {1: [3], 2: [3]}, {3: [1, 2]})

    2D example:

    >>> a = np.array([[1., 2.], [3., 4.], [-0.4, -0.8]])
    >>> get_equal_indices(a, scale_factors=[-0.4])
    ({0: [2]},)

    """

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if scale_factors is None:
        scale_factors = [1]

    a_dims = len(arr.shape)
    arr_B = arr[:, np.newaxis]

    sf_shape = tuple([len(scale_factors)] + [1] * (a_dims + 1))
    sf = np.array(scale_factors).reshape(sf_shape)

    bc = np.broadcast_arrays(arr, arr_B, sf)
    c = np.isclose(bc[0], bc[1] * bc[2])

    if a_dims > 1:
        c = np.all(c, axis=tuple(range(3, a_dims + 2)))

    out = ()
    for c_sub in c:

        w2 = np.where(c_sub)
        d = {}
        skip_idx = []

        for i in set(w2[0]):

            if i not in skip_idx:

                row_idx = np.where(w2[0] == i)[0]
                same_idx = list(w2[1][row_idx])

                if i in same_idx:

                    if len(row_idx) == 1:
                        continue

                    elif len(row_idx) > 1:
                        same_idx.remove(i)

                d.update({i: same_idx})
                skip_idx += same_idx

        out += (d,)

    return out
