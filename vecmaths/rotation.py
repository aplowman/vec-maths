"""Module containing functions related to different rotation representations."""
import numpy as np
from vecmaths.utils import prt
from vecmaths import vectors


def _vecpair2rotmat_stack(veca, vecb):
    """
    Generate stacks of rotation matrices which rotate from one set of column
    vector directions to another.

    Parameters
    ----------
    veca : ndarray
        Stacks of column three-vectors. Must have innermost shape of (3, 1).
    vecb : ndarray
        Stacks of column three-vectors. Must have innermost shape of (3, 1).

    Returns
    -------
    rot_mat : ndarray
        Stacks of rotation matrices with innermost shape of (3, 3).

    """

    if veca.shape != vecb.shape:
        raise ValueError('Shapes of `veca` and `vecb` do not '
                         'match: {} and {}'.format(veca.shape, vecb.shape))

    if veca.shape[-2:] != (3, 1):
        raise ValueError('Innermost shapes of `veca` and `vecb` '
                         'must be (3, 1), but is {}.'.format(veca.shape[-2:]))

    # Normalise
    veca = veca / np.linalg.norm(veca, axis=-2)[..., None]
    vecb = vecb / np.linalg.norm(vecb, axis=-2)[..., None]

    # Cross product matrix
    vecx = np.cross(veca, vecb, axis=-2)
    xprod = vectors.cross_prod_mat(vecx)

    # Note axis `j` has size 1:
    dot = np.einsum('...ij,...ij->...j', veca, vecb)[..., None]

    # Find which vector pairs are anti-parallel:
    neg = np.isclose(dot, -1)
    neg_idx = np.where(neg)[:-2]
    not_neg = np.logical_not(neg)
    not_neg_idx = np.where(not_neg)[:-2]

    # Find rotation matrices for anti-parallel vector pairs:
    veca_neg = veca[neg_idx]
    veca_perp = vectors.perpendicular(veca_neg, axis=-2)
    rot_mat_neg = axang2rotmat(veca_perp, 180, degrees=True, axis=-2)

    rot_mat = np.zeros_like(xprod)
    rot_mat[neg_idx] = rot_mat_neg

    # Find rotation matrices for remaining vector pairs:
    xprod_nn = xprod[not_neg_idx]
    dot_nn = dot[not_neg_idx]

    rot_mat[not_neg_idx] = np.eye(3) + xprod_nn
    rot_mat[not_neg_idx] += (xprod_nn @ xprod_nn) / (1 + dot_nn)

    return rot_mat


def vecpair2rotmat(veca, vecb, axis=0):
    """
    Generate stacks of rotation matrices which rotate from one set of vector
    direction to another.

    Parameters
    ----------
    veca : ndarray
        Array of vectors whose directions are considered the initial
        directions. Must have the same shape as `vecb`.
    vecb : ndarray
        Array of vectors whose directions are considered the final directions.
        Must have the same shape as `veca`.
    axis : int
        Axis defining the vectors.

    Returns
    -------
    rot_mat : ndarray
        Stacks of rotation matrices with innermost shape of (3, 3).

    Notes
    -----
    1.  Size-1 dimensions whose axis positions are greater than `axis` are
        squeezed out.

    """

    if veca.shape != vecb.shape:
        raise ValueError('Shapes of `veca` and `vecb` do not '
                         'match: {} and {}'.format(veca.shape, vecb.shape))

    if veca.shape[axis] != 3:
        raise ValueError('Size of dimension `axis` ({}) along `veca` and '
                         '`vecb` must be 3, but is ''{}.'.format(
                             axis, veca.shape[axis]))

    # Squeeze out size-1 dimensions after `axis`
    inner_ax_shape = np.array(list(veca.shape[axis:]))
    squeeze_axes = tuple(np.where(inner_ax_shape == 1)[0] + axis)
    veca = veca.squeeze(axis=squeeze_axes).swapaxes(-1, axis)
    vecb = vecb.squeeze(axis=squeeze_axes).swapaxes(-1, axis)

    # Ensure the innermost shape is (3, 1)
    veca = veca[..., None]
    vecb = vecb[..., None]

    return _vecpair2rotmat_stack(veca, vecb)


def axang2rotmat(rot_ax, ang, axis=0, ndim_outer=0, degrees=False):
    """
    Find the rotation matrices from axis-angle representations.

    Generated rotation matrices act on column vectors by pre-multiplication.
    Where multiple rotation matrices are generated, the returned array has
    innermost dimensions of (3, 3) corresponding to each rotation matrix.

    Parameters
    ----------
    rot_ax : ndarray
        One or more rotation axes. See Notes for details on allowed shapes.
    ang : float or list of float or ndarray
        One or more rotation angles. See Notes for details on allowed shapes.
    axis : int
        The axis of `rot_ax` corresponding to each rotation axis. This axis
        should have length three. By defaut, set to 0.
    ndim_outer : int, optional
        Determines the number of outer dimensions. The first `ndim_outer`
        dimensions of both `ax` and `ang` must have the same size. By
        default, set to 0.
    degrees : bool, optional
        If True, `ang` is interpreted as degrees, else radians. False by
        default.

    Returns
    -------
    rot_mat : ndarray of inner shape (3, 3)

    Notes
    -----
    1.  Uses the matrix form of the Rodrigues' rotation formula.

    2.  The returned rotation matrix array has an inner shape determined
        according to the input inner shapes of `rot_ax` and `angle` and the
        value of `axis`:

        `rot_ax` shape  | `axis` | `ang` shape | `rot_mat` shape
        ----------------|--------|-------------|----------------
        (3,)            | 0      | ()          | (3, 3)
        (3, 1)          | 0      |             |
        (1, 3)          | 1      |             |
        ----------------|--------|-------------|----------------
        (3,)            | 0      | (M,)        | (M, 3, 3)
        (3, 1)          | 0      |             |
        (1, 3)          | 1      |             |
        ----------------|--------|-------------|----------------
        (N, 3)          | 1      | ()          | (N, 3, 3)
        (N, 3, 1)       | 1      |             |
        (3, N)          | 0      |             |

        Additional dimensions of size 1 within the inner shape are squeezed
        out.

        In addition, an outer shape `P` (i.e. shape (p1, p2, ...)) is allowed,
        in which case `ndim_outer` should be set to the number of dimensions in
        the outer shape (i.e. `len(P)`).

    """

    ang = np.array(ang)
    inner_ang_shp = ang.shape[ndim_outer:]

    if degrees:
        ang = np.deg2rad(ang)

    # Convert to a positive axis index:
    if axis < 0:
        axis = rot_ax.ndim + axis

    if axis >= rot_ax.ndim:
        raise ValueError('`axis` must be less than the number of '
                         'dimensions in `rot_ax`.')

    if axis < ndim_outer:
        raise ValueError('`axis` cannot be less than `ndim_outer.')

    if rot_ax.shape[axis] != 3:
        raise ValueError('Specified axis of `rot_ax` ({}) must have '
                         'size 3.'.format(axis))

    outer_rot_ax_shp = rot_ax.shape[:ndim_outer]
    outer_ang_shp = ang.shape[:ndim_outer]

    if outer_rot_ax_shp != outer_ang_shp:
        msg = (
            'The first `ndim_outer` ({}) dimensions of `rot_ax` and `ang` '
            'must match, but shapes are: {} and {}.'.format(
                ndim_outer, outer_rot_ax_shp, outer_ang_shp)
        )
        raise ValueError(msg)

    # Swap axes to get the rotation axis along the final dimension
    rot_ax = np.swapaxes(rot_ax, axis, -1)

    # Squeeze out any size-one dimensions from the inner shape:
    inner_ax_dim = rot_ax.ndim - ndim_outer
    inner_ax_shape = np.array(list(rot_ax.shape[-inner_ax_dim:]))
    squeeze_axes = tuple(np.where(inner_ax_shape == 1)[0] + ndim_outer)
    rot_ax = rot_ax.squeeze(axis=squeeze_axes)

    # Ensure the innermost `rot_ax` shape is (3, 1)
    rot_ax = rot_ax[..., None]

    inner_ang_dim = ang.ndim - ndim_outer
    inner_ax_dim = rot_ax.ndim - ndim_outer
    outer_ax_shp = rot_ax.shape[:-2]

    if inner_ang_dim not in [0, 1]:
        raise ValueError('`ang` must have inner dimension of zero or one.')

    if inner_ang_dim + inner_ax_dim > 3:
        raise ValueError('Combined number of inner dimensions of `ax` and '
                         '`ang` should be two or three.')

    # Check final two dimensions of `ax` are column three-vectors:
    if rot_ax.shape[-2:] != (3, 1):
        raise ValueError('Inner shape of `ax` must be (N, 3, 1) or (3, 1) or '
                         '(3,).')

    # Normalise axis to unit vector:
    rot_ax_norm = np.linalg.norm(rot_ax, axis=-2)[..., None]
    rot_ax = rot_ax / rot_ax_norm

    ang = ang[..., None, None]

    if inner_ang_dim == 1:
        # Broadcast axis to account for multiple angles
        ax_exp = np.expand_dims(rot_ax, ndim_outer)
        ax_newshp = outer_ax_shp + inner_ang_shp + (3, 1)
        ang_newshp = outer_ax_shp + inner_ang_shp + (1, 1,)
        rot_ax = np.broadcast_to(ax_exp, ax_newshp)
        ang = np.broadcast_to(ang, ang_newshp)

    xprod = vectors.cross_prod_mat(rot_ax)
    rot_mat = np.eye(3) + np.sin(ang) * xprod
    rot_mat += (1 - np.cos(ang)) * (xprod @ xprod)

    return rot_mat


def rotmat2axang(rot_mat, degrees=False):
    """
    Decompose rotation matrices into axis-angle representations.

    Parameters
    ----------
    rot_mat : ndarray
        Stacks of rotation matrices with innermost shape (3, 3).
    degrees : bool, optional
        If True, angles are returned in degrees, otherwise in radians. Default
        is False.

    Returns
    -------
    rot_ax : ndarray
        Stacks of column vectors representing rotation axes for each rotation
        matrix. If input `rot_mat` has shape (N, M, ..., 3, 3), `rot_ax` will
        have shape (N, M, ..., 3, 1).
    angle : ndarray
        Rotation angles. If input `rot_mat` has shape (N, M, ..., 3, 3),
        `angle` will have shape (N, M, ...).

    Notes
    -----
    Following similar function in Matlab from here:
    https://github.com/marcdegraef/3Drotations/blob/master/src/MatLab/om2ax.m

    TODO:
    - Understand `P` factor in  http://doi.org/10.1088/0965-0393/23/8/083501
      and apply here if necessary.

    """

    tol = 1e-10

    # Check dimensions
    if rot_mat.shape[-2:] != (3, 3):
        raise ValueError('`rot_mat` must have inner dimensions (3, 3).')

    trc = np.trace(rot_mat, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip(0.5 * (trc - 1), -1, 1))

    if degrees:
        angle = np.rad2deg(angle)

    ax_shp = rot_mat.shape[:-2] + (3,)
    rot_ax = np.ones(ax_shp) * np.nan

    ang_zero = np.isclose(angle, 0.0)
    ang_zero_idx = np.where(ang_zero)

    if len(ang_zero_idx[0]) != 0:
        rot_ax[ang_zero_idx] = [0, 0, 1]

    # Find eigenvalues, eigenvectors for `rot_mat`
    eigval, eigvec = np.linalg.eig(rot_mat)

    # Get index of eigenvalue which is 1 + 0i
    eig_cond = np.logical_and(
        abs(np.real(eigval) - 1) < tol,
        abs(np.imag(eigval) < tol)
    )
    eig_cond_sum = np.sum(eig_cond, axis=-1)

    # Find indices where rotation axes cannot be extracted:
    no_ax = np.logical_and(
        eig_cond_sum != 1,
        np.logical_not(ang_zero)
    )
    no_ax_idx = np.where(no_ax)

    if no_ax_idx[0].size:
        no_ax_all_idx = (np.vstack(no_ax_idx).T).tolist()
        msg = ('Not exactly one eigenvector with eigenvalue of 1 found for '
               'rotation matrices at indices: {}'.format(no_ax_all_idx))
        raise ValueError(msg)

    eig_cond[ang_zero] = False
    eig_cond_idx = np.where(eig_cond)

    # Swap final axes of eigenvectors to allow indexing column vectors
    axes_comp = eigvec.swapaxes(-2, -1)[eig_cond_idx]

    # Set the axes to eigenvector with eigenvalue == 1
    rot_ax[eig_cond_idx[:-1]] = np.real(axes_comp)

    # Sort out signs of axes components:
    ax_sign_calc = np.ones_like(rot_ax)
    ax_sign_calc[..., 0] = rot_mat[..., 2, 1] - rot_mat[..., 1, 2]
    ax_sign_calc[..., 1] = rot_mat[..., 0, 2] - rot_mat[..., 2, 0]
    ax_sign_calc[..., 2] = rot_mat[..., 1, 0] - rot_mat[..., 0, 1]

    ax_sign_nz = np.logical_not(np.isclose(ax_sign_calc, 0))
    ax_sign = np.sign(ax_sign_calc[ax_sign_nz])
    rot_ax[ax_sign_nz] = np.abs(rot_ax[ax_sign_nz]) * ax_sign

    # Return axes as stacks of column vectors
    rot_ax = rot_ax[..., None]

    return (rot_ax, angle)


def align_xy(box):
    """Align a parallelepiped such that the first edge vector is along x and
    the second is in the xy-plane.

    Parameters
    ----------
    box : ndarray of shape (3, 3)
        Array of column vectors representing edge vectors of a parallelepiped.

    Returns
    -------
    aligned_box : ndarray of shape (3, 3)
        Array of column vectors representing edge vectors of rotated
        parallelepiped.

    """

    veca = box[:, 0]
    vecb = box[:, 1]
    vecc = box[:, 2]

    amag, bmag, cmag = [np.linalg.norm(i) for i in [veca, vecb, vecc]]

    cos_gamma = vectors.vecpair_cos(veca, vecb)
    sin_gamma = vectors.vecpair_sin(veca, vecb)
    cos_beta = vectors.vecpair_cos(vecc, veca)
    bc_dot = np.einsum('i,i->', vecb, vecc)

    b_x = bmag * cos_gamma
    b_y = bmag * sin_gamma
    c_x = cmag * cos_beta
    c_y = (bc_dot - (b_x * c_x)) / b_y

    veca_new = np.array([amag, 0, 0])
    vecb_new = np.array([b_x, b_y, 0])
    vecc_new = np.array([c_x, c_y, np.sqrt(cmag**2 - c_x**2 - c_y**2)])

    aligned_box = np.vstack([veca_new, vecb_new, vecc_new]).T

    return aligned_box
