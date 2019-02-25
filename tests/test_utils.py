"""Module containing unit tests on the `vecmaths.utils` module."""

import unittest
import numpy as np
from vecmaths.utils import validate_array_args, get_half_space_flip_idx


class ValidateArrayArgsTestCase(unittest.TestCase):
    """Tests on the `validate_array_args` function."""

    def test_raises_on_bad_shape(self):
        """Test ValueError raised when a single array does not have
        the expected shape."""

        my_arr = np.random.randint(0, 9, (3, 2))

        with self.assertRaises(ValueError):
            validate_array_args(('my_arr', my_arr, (2, 2)))

    def test_no_raise_on_good_shape(self):
        """Test no return/raise when an array has the expected shape."""

        shp = (3, 2)
        my_arr = np.random.randint(0, 9, shp)

        self.assertIsNone(validate_array_args(('my_arr', my_arr, shp)))

    def test_raises_on_inconsistent_symbol_single(self):
        """Test ValueError raised when a symbol is associated with two distinct
        dimension lengths within a single array."""

        expected_shp = ('N', 'N', 3)
        actual_shp = (2, 3, 3)
        my_arr = np.random.randint(0, 9, actual_shp)

        with self.assertRaises(ValueError):
            validate_array_args(('my_arr', my_arr, expected_shp))

    def test_raises_on_inconsistent_symbol_multi(self):
        """Test ValueError raised when a symbol is associated with two distinct
        dimension lengths across two arrays."""

        expected_shp_1 = ('N', 2)
        actual_shp_1 = (2, 2)

        expected_shp_2 = (3, 'N')
        actual_shp_2 = (3, 5)

        my_arr_1 = np.random.randint(0, 9, actual_shp_1)
        my_arr_2 = np.random.randint(0, 9, actual_shp_2)

        with self.assertRaises(ValueError):
            validate_array_args(
                ('my_arr_1', my_arr_1, expected_shp_1),
                ('my_arr_2', my_arr_2, expected_shp_2)
            )


class HalfSpaceFlipTestCase(unittest.TestCase):
    """Tests on the `utils.get_half_space_flip_idx` function."""

    def test_raises_on_bad_shape(self):
        """Test ValueError raised for unexpected array shapes."""

        good_shapes = [
            (3, 1),
            (4, 3, 1),
            (2, 4, 3, 1),
        ]
        bad_shapes = [
            (1,),
            (3,),
            (1, 3),
            (4, 1, 3),
            (2, 4, 1, 3)
        ]

        for i in good_shapes:
            get_half_space_flip_idx(np.random.random(i))

        with self.assertRaises(ValueError):
            for i in bad_shapes:
                get_half_space_flip_idx(np.random.random(i))

    def test_all_quadrants_correct_flip(self):
        """Test vectors in all quadrants flip as expected."""

        expected_flip_x_lt_0 = [
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, -1],
            [-1, 1, 0],
            [-1, 1, 1]
        ]

        expected_flip_x_0_y_lt_0 = [
            [0, -1, -1],
            [0, -1, 0],
            [0, -1, 1]
        ]

        expected_flip_x_0_y_0_z_lt_0 = [
            [0, 0, -1]
        ]

        expected_no_flip = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, -1, -1],
            [1, -1, 0],
            [1, -1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 1, 0],
            [1, 1, 1],
        ]

        all_vecs = np.array(
            expected_flip_x_lt_0 +
            expected_flip_x_0_y_lt_0 +
            expected_flip_x_0_y_0_z_lt_0 +
            expected_no_flip
        )[:, :, None]

        expected_flip_idx = np.arange(13)
        flip_idx = get_half_space_flip_idx(all_vecs)
        self.assertTrue(np.all(flip_idx == expected_flip_idx))
