"""Module defining unit tests on the vecutils.vectors module."""

import unittest
import numpy as np
from vecmaths import vectors


class PerpendicularTestCase(unittest.TestCase):
    """
    Tests on the function `vectors.perpendicular`.

    """

    def test_is_perpendicular(self):
        """
        Test for a stack of vectors, perpendicular vectors are found.

        """
        num_in = 100
        vecs_in = np.random.random((num_in, 3))
        vecs_out = vectors.perpendicular(vecs_in, axis=-1)
        dot = np.einsum('ij,ij->i', vecs_in, vecs_out)
        self.assertTrue(np.allclose(dot, np.zeros(num_in,)))
