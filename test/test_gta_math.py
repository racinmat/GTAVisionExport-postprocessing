import unittest
import numpy as np
from unittest_data_provider import data_provider
from gta_math import construct_model_matrix, create_model_rot_matrix, rot_matrix_to_euler_angles, create_rot_matrix, \
    model_rot_matrix_to_euler_angles, car_and_relative_cam_to_absolute_cam_rotation_matrix, \
    relative_and_absolute_camera_to_car_rotation_matrix, car_and_relative_cam_to_absolute_cam_position
from toyota import get_my_car_position_and_rotation


def euler_angles_matrix_data():
    return (
        ([0., 0., 0.],),
        ([0., 90., 0.],),
        ([0., 0., 90.],),
        ([90., 0., 0.],),
        ([40., 50., 60.],),
        ([20., 30., 0.],),
    )


def car_to_camera_view_data():
    return (
        (
            [-0.006126282270997763, 0.84723699092864990, 0.44408929347991943],
            [-0.006126282270997763, 0.84723699092864990, 0.44408929347991943],
            [0., 0., 0.]
        ),
        (
            [-0.006126282270997763, 0.84723699092864990, 0.44408929347991943],
            [0.84683728218078610, 0.007344349753111601, 90.41197204589844000],
            [0., 0., 90.]
        ),
        (
            [-0.006126282270997763, 0.84723699092864990, 0.44408929347991943],
            [0.008536399342119694, -0.84658384323120120, -179.62216186523438000],
            [0., 0., 180.]
        ),
        (
            [-0.006126282270997763, 0.84723699092864990, 0.44408929347991943],
            [-0.84574091434478760, -0.011379104107618332, -89.68127441406250000],
            [0., 0., 270.]
        ),
    )


def car_to_camera_position_data():
    return (
        (
            [2524.95751953125000000, 3333.01220703125000000, 52.35118103027344000],
            [-1.26793634891510000, -4.02545595169067400, -0.25146627426147460],
            [-0.05999999865889549, 1.50000000000000000, 1.07649993896484380],
            [2524.83105468750000000, 3334.53613281250000000, 53.38750457763672000],
        ),
        (
            [2524.95751953125000000, 3333.01220703125000000, 52.35118103027344000],
            [-1.26793634891510000, -4.02545595169067400, -0.25146627426147460],
            [15.93999958038330000, 17.50000000000000000, 6.07649993896484400],
            [2540.50146484375000000, 3350.60375976562500000, 59.15400695800781000],
        ),
        (
            [2524.95751953125000000, 3333.01220703125000000, 52.35118103027344000],
            [-1.26793634891510000, -4.02545595169067400, -0.25146627426147460],
            [-0.05999999865889549, 33.50000000000000000, 6.07649993896484400],
            [2524.58129882812500000, 3366.63574218750000000, 57.69403457641601600],
        ),
        (
            [2524.95751953125000000, 3333.01220703125000000, 52.35118103027344000],
            [-1.26793634891510000, -4.02545595169067400, -0.25146627426147460],
            [-16.05999946594238300, 17.50000000000000000, 6.07649993896484400],
            [2508.55883789062500000, 3350.65502929687500000, 56.91956329345703000],
        ),
        (
            [2524.95751953125000000, 3333.01220703125000000, 52.35118103027344000],
            [-1.26793634891510000, -4.02545595169067400, -0.25146627426147460],
            [0.48000001907348633, 1.50000000000000000, 1.07649993896484380],
            [2525.36669921875000000, 3334.53466796875000000, 53.43165969848633000],
        ),
        (
            [2524.95751953125000000, 3333.01220703125000000, 52.35118103027344000],
            [-1.26793634891510000, -4.02545595169067400, -0.25146627426147460],
            [-0.05999999865889549, 17.50000000000000000, 25.07649993896484400],
            [2523.16894531250000000, 3351.02416992187500000, 77.00225830078125000],
        ),
    )


class TestGtaMath(unittest.TestCase):

    @data_provider(euler_angles_matrix_data)
    def test_model_euler_angles_matrix_transformation(self, data):
        orig_angles = np.array(data)
        matrix = create_model_rot_matrix(orig_angles)
        angles = model_rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    @data_provider(euler_angles_matrix_data)
    def test_euler_angles_matrix_transformation(self, data):
        orig_angles = np.array(data)
        matrix = create_rot_matrix(orig_angles)
        angles = rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    @data_provider(car_to_camera_view_data)
    def test_car_to_camera_view_rotation(self, car_rot_data, cam_rot_data, rel_cam_rot_data):
        correct_car_rot = np.array(car_rot_data)
        cam_rot = np.array(cam_rot_data)
        cam_rel_rot = np.array(rel_cam_rot_data)
        orig_cam_rot_m = create_rot_matrix(cam_rot)
        cam_rot_m = car_and_relative_cam_to_absolute_cam_rotation_matrix(correct_car_rot, cam_rel_rot)
        self.assertTrue(np.allclose(cam_rot_m, orig_cam_rot_m, atol=3.e-3))

    @data_provider(car_to_camera_view_data)
    def test_camera_to_car_rotation(self, car_rot_data, cam_rot_data, rel_cam_rot_data):
        cam_rot = np.array(cam_rot_data)
        cam_rel_rot = np.array(rel_cam_rot_data)
        orig_car_rot_m = create_model_rot_matrix(np.array(car_rot_data))
        car_rot_m = relative_and_absolute_camera_to_car_rotation_matrix(cam_rot, cam_rel_rot)
        self.assertTrue(np.allclose(car_rot_m, orig_car_rot_m, atol=3.e-3))

    @data_provider(car_to_camera_position_data)
    def test_car_to_camera_position(self, car_pos_data, car_rot_data, rel_cam_pos_data, cam_pos_data):
        car_pos = np.array(car_pos_data)
        car_rot = np.array(car_rot_data)
        cam_rel_pos = np.array(rel_cam_pos_data)
        orig_cam_pos = np.array(cam_pos_data)
        cam_pos = car_and_relative_cam_to_absolute_cam_position(car_pos, car_rot, cam_rel_pos)
        self.assertTrue(np.allclose(cam_pos, orig_cam_pos, atol=1.e-1))


if __name__ == '__main__':
    unittest.main()
