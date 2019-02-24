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
