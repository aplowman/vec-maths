"""Module containing unit tests on the `vecmaths.rotation` module."""

import unittest
import copy
import numpy as np
from vecmaths import rotation
from vecmaths import vectors
from vecmaths.utils import prt


class RotMatAxAngConsistencyTestCase(unittest.TestCase):
    """
    Tests on consistency between rotation matrix and axis-angle
    representations.

    """

    def test_axis_angle_consistent(self):
        """
        Test conversion from axis-angle -> matrix -> axis-angle with positive
        angles are consistent.

        """
        axis_1 = np.random.random((3, 1))
        axis_1_norm = axis_1 / np.linalg.norm(axis_1, axis=0)
        ang_1 = np.random.random() * 180
        params = {
            'rot_ax': axis_1,
            'ang': ang_1,
            'axis': 0,
            'ndim_outer': 0,
            'degrees': True,
        }
        rot_mat = rotation.axang2rotmat(**params)
        axis_2, ang_2 = rotation.rotmat2axang(rot_mat, degrees=True)
        axis_2_norm = axis_2 / np.linalg.norm(axis_2, axis=0)

        axis_equal = np.allclose(axis_1_norm, axis_2_norm)
        angle_equal = np.isclose(ang_1, ang_2)

        self.assertTrue(axis_equal and angle_equal)

    def test_always_positive_angle(self):
        """
        Test conversion from axis-angle -> matrix -> axis-angle with negative
        angle flips the axis and angle signs, so the output angle is positive.

        """
        axis_1 = np.random.random((3, 1))
        axis_1_norm = axis_1 / np.linalg.norm(axis_1, axis=0)
        ang_1 = (np.random.random() - 1) * 180
        params = {
            'rot_ax': axis_1,
            'ang': ang_1,
            'axis': 0,
            'ndim_outer': 0,
            'degrees': True,
        }
        rot_mat = rotation.axang2rotmat(**params)
        axis_2, ang_2 = rotation.rotmat2axang(rot_mat, degrees=True)
        axis_2_norm = axis_2 / np.linalg.norm(axis_2, axis=0)

        axis_equal = np.allclose(axis_1_norm, -axis_2_norm)
        angle_equal = np.isclose(ang_1, -ang_2)

        self.assertTrue(axis_equal and angle_equal)


class RotMat2AxAngTestCase(unittest.TestCase):
    """Unit tests on the function `rotation.rotmat2axang`."""

    def test_known_axis_angle(self):
        """
        Test the generated rotation matrix for a known rotation (90 degrees
        about the z-axis) is correct.

        """
        rot = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        axis, ang = rotation.rotmat2axang(rot)
        known_axis = np.array([[0, 0, 1]]).T
        known_ang = np.pi / 2

        axis_equal = np.allclose(axis, known_axis)
        angle_equal = np.isclose(ang, known_ang)

        self.assertTrue(axis_equal and angle_equal)


class AxAng2RotMatTestCase(unittest.TestCase):
    """Unit tests on the function `rotation.axang2rotmat`."""

    def test_known_rotation(self):
        """
        Test the generated rotation matrix for a known rotation (90 degrees
        about the z-axis) is correct.

        """
        axis = np.array([0, 0, 1])
        ang = np.pi / 2
        rot = rotation.axang2rotmat(axis, ang)
        known_rot = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        self.assertTrue(np.allclose(rot, known_rot))

    def test_degrees_to_radians(self):
        """
        Test indentical rotation matrices are generated whether angles are
        input in degrees or radians.

        """
        axis = np.random.random((3,))
        ang_deg = np.random.random() * 180
        ang_rad = np.deg2rad(ang_deg)
        rot_deg = rotation.axang2rotmat(axis, ang_deg, degrees=True)
        rot_rad = rotation.axang2rotmat(axis, ang_rad, degrees=False)
        self.assertTrue(np.allclose(rot_deg, rot_rad))

    def test_zero_rotation(self):
        """
        Test a rotation about a random axis by zero degrees generates the
        identity matrix.

        """
        params = {
            'rot_ax': np.random.random((3,)),
            'ang': 0,
            'axis': 0,
            'ndim_outer': 0,
            'degrees': True,
        }
        rot_mat = rotation.axang2rotmat(**params)
        self.assertTrue(np.allclose(rot_mat, np.eye(3)))

    def test_broadcast_axes(self):
        """
        Test a single axis can be broadcast correctly over multiple angles.

        """

        params_multiple = {
            'rot_ax': np.random.random((3,)),
            'ang': (np.random.random((2,)) - 0.5) * 180,
            'axis': 0,
            'ndim_outer': 0,
            'degrees': True,
        }
        params_single = copy.deepcopy(params_multiple)
        params_single['ang'] = params_single['ang'][0]

        rot_mat_mult = rotation.axang2rotmat(**params_multiple)
        rot_mat_sing = rotation.axang2rotmat(**params_single)

        self.assertTrue(np.allclose(rot_mat_mult[0], rot_mat_sing))

    def test_broadcast_angles(self):
        """
        Test a single angle can be broadcast correctly over multiple axes.

        """
        params_multiple = {
            'rot_ax': np.random.random((2, 3)),
            'ang': (np.random.random() - 0.5) * 180,
            'axis': 1,
            'ndim_outer': 0,
            'degrees': True,
        }

        params_single = copy.deepcopy(params_multiple)
        params_single['rot_ax'] = params_single['rot_ax'][0]
        params_single['axis'] = 0

        rot_mat_mult = rotation.axang2rotmat(**params_multiple)
        rot_mat_sing = rotation.axang2rotmat(**params_single)

        self.assertTrue(np.allclose(rot_mat_mult[0], rot_mat_sing))

    def test_outer_shape(self):
        """
        Test assigning `ndim_outer` != 0 is implemented correctly.

        """
        # Outer shape (shared by axes and angles):
        out_shp = (3, 2)

        axes_multiple = np.random.random(out_shp + (3,))
        ang_multiple = (np.random.random(out_shp) - 0.5) * 180
        params_multiple = {
            'rot_ax': axes_multiple,
            'ang': ang_multiple,
            'axis': -1,
            'ndim_outer': len(out_shp),
            'degrees': True,
        }
        params_single = {
            'rot_ax': axes_multiple[0, 0],
            'ang': ang_multiple[0, 0],
            'axis': -1,
            'ndim_outer': 0,
            'degrees': True,
        }
        rot_mat_mult = rotation.axang2rotmat(**params_multiple)
        rot_mat_sing = rotation.axang2rotmat(**params_single)

        self.assertTrue(np.allclose(rot_mat_mult[0, 0], rot_mat_sing))


class AlignXYTestCase(unittest.TestCase):
    """Tests on function `rotation.align_xy`"""

    def test_alignment(self):
        """Check first vector is aligned in the x-direction and second vector
        is aligned in the xy-plane.

        """
        box = np.random.random((3, 3))
        box_aligned = rotation.align_xy(box)
        self.assertTrue(np.allclose(box_aligned[[1, 2, 2], [0, 0, 1]], 0))

    def test_consistent_volume(self):
        """Check volume of box remains the same after alignment."""

        def get_volume(box):
            """Get the volume of a parallelepiped."""
            signed_vol = np.dot(np.cross(box[:, 0], box[:, 1]), box[:, 2])
            vol = np.linalg.norm(signed_vol)
            return vol

        box = np.random.random((3, 3))
        vol = get_volume(box)
        box_aligned = rotation.align_xy(box)
        vol_aligned = get_volume(box_aligned)

        self.assertAlmostEqual(vol, vol_aligned)

    def test_edge_lengths(self):
        """Check edge lengths of box remain the same after alignment."""

        box = np.random.random((3, 3))
        box_aligned = rotation.align_xy(box)

        box_mag = np.linalg.norm(box, axis=0)
        box_aligned_mag = np.linalg.norm(box_aligned, axis=0)

        self.assertTrue(np.allclose(box_mag, box_aligned_mag))

    def test_internal_angles(self):
        """Check internal angles of box remain the same after alignment."""

        def get_internal_angles(box):
            """Get internal angles of a box."""
            ang = vectors.vecpair_angle(box[:, [0, 0, 1]],
                                        box[:, [1, 2, 2]], axis=0)
            return ang

        box = np.random.random((3, 3))
        box_aligned = rotation.align_xy(box)

        ang = get_internal_angles(box)
        ang_aligned = get_internal_angles(box_aligned)

        self.assertTrue(np.allclose(ang, ang_aligned))


class VecPair2RotMatTestCase(unittest.TestCase):
    """Tests on function `rotation.vecpair2rotmat`."""

    def test_single_anti_parallel_valid(self):
        """Test non-nan output for a single, known anti-parallel pair."""

        veca = np.array([0, 0, 1])
        vecb = np.array([0, 0, -1])
        rot_mat = rotation.vecpair2rotmat(veca, vecb)
        self.assertFalse(np.any(np.isnan(rot_mat)))
