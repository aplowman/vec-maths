"""Module containing general utility functions."""

import numpy as np


def prt(obj, name, slc=None):
    """
    Print an object, with it's size and dtype if it's an ndarray.

    Helpful for debugging.

    Parameters
    ----------
    obj : object
        Object to print.
    name : str
        Name of object to print.
    slc : tuple of slice object, optional
        If not None, only print the sub array defined by the slice.
    """

    if isinstance(obj, np.ndarray):

        if slc is not None:
            slc_str = ', '.join(['{}:{}:{}'.format(
                s.start or '', s.stop or '', s.step or '') for s in slc])
            slc_str = '[' + slc_str + ']'
        else:
            slc_str = ''

        fmt_str = '{} {} {} {}: \n{}\n'
        fmts = (name, obj.shape, slc_str,
                obj.dtype, obj[slc] if slc is not None else obj)

    elif isinstance(obj, dict):

        fmt_str = ''
        fmts = []
        for key, val in sorted(obj.items()):
            fmt_str += '{}: {}\n'
            fmts.extend([key, val])

    else:

        fmt_str = '{} ({}): \n{}\n'
        fmts = (name, type(obj), obj)

    print(fmt_str.format(*fmts))


def snap_arr(arr, val, tol):
    """Snap array elements within a certain value.

    Parameters
    ----------
    arr : ndarray
    val : float
        Value to snap to.
    tol : float
        Array elements within this tolerance of `val` will be snapped to `val`.

    Returns
    -------
    arr_snapped : ndarray
        Copy of input array, where elements which are close to `val` are set to
        be exactly `val` if within tolerance `tol`.

    """

    arr_snapped = np.copy(arr)
    arr_snapped[abs(arr - val) < tol] = val
    return arr_snapped


def validate_array_args(*args):
    """Validate the shapes of a set of Numpy arrays.

    Parameters
    ----------
    arg : tuple of (str, ndarray, tuple of (str or int))
        Each arg has: the name of a Numpy array, the array itself, and the 
        expected shape of the array. The expected shape is passed as a tuple,
        whose elements maybe integers of strings; if a string, the element
        represent a symbol used to represent the shape in that dimension.

    Notes
    -----
    This function also checks that, if symbols have been repeated in the
    elements of expected array shapes for multiple arrays, those dimension
    sizes are consistent across the different arrays.

    Returns
    -------
    None

    Example
    -------

    """

    symbols = {}
    arr_msg = ('`{}` must be a Numpy array of shape {}')
    multi_msg = ('Inconsistent dimension size for symbol: {}')

    for i in args:

        name = i[0]
        arr = i[1]
        shape = i[2]

        if not isinstance(arr, np.ndarray) or (arr.ndim != len(shape)):
            raise ValueError(arr_msg.format(name, shape))

        for i_idx, i in enumerate(shape):

            if isinstance(i, str):
                if i not in symbols:
                    symbols[i] = arr.shape[i_idx]
                else:
                    if arr.shape[i_idx] != symbols[i]:
                        raise ValueError(multi_msg.format(symbols[i]))
                continue

            if arr.shape[i_idx] != i:
                raise ValueError(arr_msg)


def get_half_space_flip_idx(vecs):
    """Get the indices of vectors that, if multiplied by minus one, would
    ensure no anti-parallel vectors in the set.

    Parameters
    ----------
    vecs : ndarray of inner shape (3, 1)

    Returns
    -------
    flip_idx : ndarray

    Notes
    -----
    For a set of vectors in R3, we might want to map all vectors such that no
    two are anti-parallel. This function returns, for a set of vectors, the
    indices of vectors that require flipping (i.e. multiplying by -1) in order
    to ensure the whole set of vectors is contained within the positive-x half-
    space.

    Specifically, the conditions for flipping are:
        - [x-component < 0] OR
        - [(x-component == 0) and (y-component < 0)] OR
        - [(x-component == 0) and (y-component == 0) and (z-component < 0)]

    """

    in_shp = vecs.shape[-2:]
    if in_shp != (3, 1):
        msg = ('Input array must have inner shape (3, 1), '
               'but has inner shape: {}'.format(in_shp))
        raise ValueError(msg)

    a = vecs[..., 0, 0]
    b = vecs[..., 1, 0]
    c = vecs[..., 2, 0]

    flip_cond = np.logical_or(
        np.logical_or(
            a < 0,
            np.logical_and(np.isclose(a, 0), b < 0),
        ),
        np.logical_and(
            np.logical_and(np.isclose(a, 0), np.isclose(b, 0)),
            c < 0
        )
    )
    flip_idx = np.where(flip_cond)

    return flip_idx
