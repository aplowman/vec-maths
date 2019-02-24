"""Module containing unit tests on the `vecmaths.vectors` module."""

import unittest
import numpy as np
from vecmaths import vectors


class NormaliseTestCase(unittest.TestCase):
    """Tests on the function `vectors.normalise`."""

    def test_normalised_vectors_unit(self):
        """Test normalised vectors have magnitude one."""
        num_vecs = 2
        vecs = np.random.random((num_vecs, 3))
        vecs_normd = vectors.normalise(vecs)
        vecs_normed_norm = np.linalg.norm(vecs_normd, axis=-1)
        self.assertTrue(np.allclose(vecs_normed_norm, 1))

    def test_zero_vectors(self):
        """Test zero vectors are not modified by normalisation."""
        num_vecs = 2
        vecs = np.random.random((num_vecs, 3))
        zero_idx = 1
        vecs[zero_idx] = 0
        vecs_normd = vectors.normalise(vecs)
        self.assertTrue(np.allclose(vecs[zero_idx], vecs_normd[zero_idx]))

    def test_big_shape(self):
        """Test normalisation of an array with a larger outer shape"""
        shp = (4, 2, 3)
        check_idx = (1, 0)
        vecs = np.random.random(shp)
        vecs_normd = vectors.normalise(vecs)
        check_vec = vecs[check_idx]
        check_vec_n = np.linalg.norm(check_vec)
        check_vec_normd = check_vec / check_vec_n
        self.assertTrue(np.allclose(vecs_normd[check_idx], check_vec_normd))

    def test_equal_shape(self):
        """Test normalisation does not alter the shape of the array."""
        shp = (4, 2, 3)
        vecs = np.random.random(shp)
        vecs_normd = vectors.normalise(vecs)
        self.assertTrue(vecs_normd.shape == shp)


class PerpendicularTestCase(unittest.TestCase):
    """Tests on the function `vectors.perpendicular`."""

    def test_is_perpendicular(self):
        """Test for a stack of vectors, perpendicular vectors are found."""
        num_in = 100
        vecs_in = np.random.random((num_in, 3))
        vecs_out = vectors.perpendicular(vecs_in, axis=-1)
        dot = np.einsum('ij,ij->i', vecs_in, vecs_out)
        self.assertTrue(np.allclose(dot, np.zeros(num_in,)))


class VecPairCosTestCase(unittest.TestCase):
    """Tests on function `vectors.vecpair_cos`."""

    def test_vec_pair_equal_cos(self):
        """Test cosine between equal vectors is 1."""

        veca = np.random.random(3,)
        vecb = veca
        cos = vectors.vecpair_cos(veca, vecb)
        self.assertTrue(np.isclose(cos, 1))

    def test_vec_pair_opposite_cos(self):
        """Test cosine between opposite vectors is -1."""

        veca = np.random.random(3,)
        vecb = -veca
        cos = vectors.vecpair_cos(veca, vecb)
        self.assertTrue(np.isclose(cos, -1))

    def test_vec_pair_perpendicular_cos(self):
        """Test cosine between perpendicular vectors is 0."""

        veca = np.array([1, 0, 0])
        vecb = np.array([0, 1, 0])
        cos = vectors.vecpair_cos(veca, vecb)
        self.assertTrue(np.isclose(cos, 0))

    def test_normalisation(self):
        """Are normalised/non-normalised vector pairs equivalent?"""

        veca = np.random.random(3,)
        vecb = np.random.random(3,)
        veca_normd = veca / np.linalg.norm(veca)
        vecb_normd = vecb / np.linalg.norm(vecb)
        cos = vectors.vecpair_cos(veca, vecb)
        cos_normd_vecs = vectors.vecpair_cos(veca_normd, vecb_normd)
        self.assertTrue(np.isclose(cos, cos_normd_vecs))


class VecPairSinTestCase(unittest.TestCase):
    """Tests on function `vectors.vecpair_sin`."""

    def test_vec_pair_equal_sin(self):
        """Test sine between equal vectors is 0."""

        veca = np.random.random(3,)
        vecb = veca
        sin = vectors.vecpair_sin(veca, vecb)
        self.assertTrue(np.isclose(sin, 0))

    def test_vec_pair_opposite_sin(self):
        """Test sine between opposite vectors is 0."""

        veca = np.random.random(3,)
        vecb = -veca
        sin = vectors.vecpair_sin(veca, vecb)
        self.assertTrue(np.isclose(sin, 0))

    def test_vec_pair_perpendicular_sin(self):
        """Test sine between perpendicular vectors is 1."""

        veca = np.array([1, 0, 0])
        vecb = np.array([0, 1, 0])
        sin = vectors.vecpair_sin(veca, vecb)
        self.assertTrue(np.isclose(sin, 1))

    def test_normalisation(self):
        """Are normalised/non-normalised vector pairs equivalent?"""

        veca = np.random.random(3,)
        vecb = np.random.random(3,)
        veca_normd = veca / np.linalg.norm(veca)
        vecb_normd = vecb / np.linalg.norm(vecb)
        sin = vectors.vecpair_sin(veca, vecb)
        sin_normd_vecs = vectors.vecpair_sin(veca_normd, vecb_normd)
        self.assertTrue(np.isclose(sin, sin_normd_vecs))


class VecPairAngleTestCase(unittest.TestCase):
    """Tests on function `vectors.vecpair_angle`."""

    def test_known_angle(self):
        """Test known angle correctly computed for two vectors."""
        veca = np.array([0, 0, 1])
        vecb = np.array([0, 1, 0])
        ang = vectors.vecpair_angle(veca, vecb, degrees=True)
        self.assertTrue(np.isclose(ang, 90))

    def test_zero_angle(self):
        """Test angle of zero found between equivalent vectors."""
        veca = np.array([0, 0, 1])
        vecb = np.array([0, 0, 1])
        ang = vectors.vecpair_angle(veca, vecb, degrees=True)
        self.assertTrue(np.isclose(ang, 0))


class FindNonParallelIntVecsTestCase(unittest.TestCase):
    """Test case for `vectors.find_non_parallel_int_vecs`."""

    def test_search_size_components(self):
        """Check the maximum vector component is equal to the search size."""
        search_size = 5
        v = vectors.find_non_parallel_int_vecs(search_size, tile=False)
        self.assertEqual(np.max(v), search_size)

    def test_invalid_search_size(self):
        """Test exceptions are raised for invalid search sizes."""
        invalid_ss = range(-2, 1)
        for i in invalid_ss:
            with self.assertRaises(ValueError):
                vectors.find_non_parallel_int_vecs(i)

    def test_non_parallel_tiled(self):
        """Test for a search size of five with `tile` True, found vectors are
        non (anti-)parallel.

        """
        v = vectors.find_non_parallel_int_vecs(5, tile=True)

        # Find cross product of each vector with all vectors
        cross_self = np.cross(v, v[:, np.newaxis])
        cross_self_zero = np.all(cross_self == 0, axis=2)
        self.assertEqual(np.max(np.sum(cross_self_zero, axis=1)), 1)

    def test_non_parallel(self):
        """Test for a search size of five with `tile` False found vectors are
        non (anti-)parallel.

        """
        v = vectors.find_non_parallel_int_vecs(5, tile=False)

        # Find cross product of each vector with all vectors
        cross_self = np.cross(v, v[:, np.newaxis])
        cross_self_zero = np.all(cross_self == 0, axis=2)
        self.assertEqual(np.max(np.sum(cross_self_zero, axis=1)), 1)
