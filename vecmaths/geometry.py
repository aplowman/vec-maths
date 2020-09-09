"""Module defining functions related to geometry."""
import numpy as np


def get_box_corners(box, origin=None):
    """
    Get all 8 corners of parallelopipeds, each defined by three edge vectors.

    Parameters
    ----------
    box : ndarray of shape (N, 3, 3) or (3, 3)
        Array defining N parallelopipeds, each as three 3D column vectors which
        define the edges of the parallelopipeds.
    origin : ndarray of shape (3, N), optional
        Array defining the N origins of N parallelopipeds as 3D column vectors.

    Returns
    -------
    ndarray of shape (N, 3, 8)
        Returns 8 3D column vectors for each input parallelopiped.

    TODO:
    -   vectorise properly with axis keyword.

    Examples
    --------
    >>> a = np.random.randint(-1, 4, (2, 3, 3))
    >>> a
    [[[ 0  3  1]
      [ 2 -1 -1]
      [ 1  2  0]]

     [[ 0  0  3]
      [ 1  2  0]
      [-1  1 -1]]]
    >>> geometry.get_box_corners(a)
    array([[[ 0.,  0.,  3.,  1.,  3.,  1.,  4.,  4.],
            [ 0.,  2., -1., -1.,  1.,  1., -2.,  0.],
            [ 0.,  1.,  2.,  0.,  3.,  1.,  2.,  3.]],

           [[ 0.,  0.,  0.,  3.,  0.,  3.,  3.,  3.],
            [ 0.,  1.,  2.,  0.,  3.,  1.,  2.,  3.],
            [ 0., -1.,  1., -1.,  0., -2.,  0., -1.]]])

    """

    if box.ndim == 2:
        box = box[np.newaxis]

    num_boxes = box.shape[0]

    if origin is None:
        origin = np.zeros((3, num_boxes), dtype=box.dtype)

    corners = np.zeros((num_boxes, 3, 8), dtype=box.dtype)
    corners[:, :, 1] = box[:, :, 0]
    corners[:, :, 2] = box[:, :, 1]
    corners[:, :, 3] = box[:, :, 2]
    corners[:, :, 4] = box[:, :, 0] + box[:, :, 1]
    corners[:, :, 5] = box[:, :, 0] + box[:, :, 2]
    corners[:, :, 6] = box[:, :, 1] + box[:, :, 2]
    corners[:, :, 7] = box[:, :, 0] + box[:, :, 1] + box[:, :, 2]

    corners += origin.T[:, :, np.newaxis]

    return corners


def get_box_xyz(box, origin=None, faces=False):
    """
    Get coordinates of paths which trace the edges of parallelopipeds
    defined by edge vectors and origins. Useful for plotting parallelopipeds.

    Parameters
    ----------
    box : ndarray of shape (N, 3, 3) or (3, 3)
        Array defining N parallelopipeds, each as three 3D column vectors which
        define the edges of the parallelopipeds.
    origin : ndarray of shape (3, N) or (3,)
        Array defining the N origins of N parallelopipeds as 3D column vectors.
    faces : bool, optional
        If False, returns an array of shape (N, 3, 30) where the coordinates of
        a path tracing the edges of each of N parallelopipeds are returned as
        column 30 vectors.

        If True, returns a dict where the coordinates for
        each face is a key value pair. Keys are like `face01a`, where the
        numbers refer to the column indices of the vectors in the plane of the
        face to plot, the `a` faces intersect the origin and the `b` faces are
        parallel to the `a` faces. Values are arrays of shape (N, 3, 5), which
        define the coordinates of a given face as five 3D column vectors for
        each of the N input parallelopipeds.

    Returns
    -------
    ndarray of shape (N, 3, 30) or dict of str : ndarray of shape (N, 3, 5)
    (see `faces` parameter).

    """

    if box.ndim == 2:
        box = box[np.newaxis]

    num_boxes = box.shape[0]

    if origin is None:
        origin = np.zeros((3, num_boxes), dtype=box.dtype)

    elif origin.ndim == 1:
        origin = origin[:, np.newaxis]

    if origin.shape[1] != box.shape[0]:
        raise ValueError('If `origin` is specified, there must be an origin '
                         'specified for each box.')

    corns = get_box_corners(box, origin=origin)

    face01a = corns[:, :, [0, 1, 4, 2, 0]]
    face01b = corns[:, :, [3, 5, 7, 6, 3]]
    face02a = corns[:, :, [0, 1, 5, 3, 0]]
    face02b = corns[:, :, [2, 4, 7, 6, 2]]
    face12a = corns[:, :, [0, 2, 6, 3, 0]]
    face12b = corns[:, :, [1, 4, 7, 5, 1]]

    coords = [face01a, face01b, face02a, face02b, face12a, face12b]

    if not faces:
        xyz = np.concatenate(coords, axis=2)

    else:
        face_names = ['face01a', 'face01b', 'face02a',
                      'face02b', 'face12a', 'face12b']
        xyz = dict(zip(face_names, coords))

    return xyz

def polar2cart_2D(r, θ):
    """
    Convert 2D polar coordinates to Cartesian coordinates.

    """

    x = r * np.cos(θ)
    y = r * np.sin(θ)

    return x, y

