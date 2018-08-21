import unittest
import numpy as np
from unittest_data_provider import data_provider
from gta_math import construct_model_matrix, create_model_rot_matrix, rot_matrix_to_euler_angles, create_rot_matrix, \
    model_rot_matrix_to_euler_angles, car_and_relative_cam_to_absolute_cam_rotation_matrix, \
    relative_and_absolute_camera_to_car_rotation_matrix
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
        # (
        #     [-0.006126282270997763, 0.84723699092864990, 0.44408929347991943],
        #     [-0.006126282270997763, 0.84723699092864990, 0.44408929347991943],
        #     [0., 0., 0.]
        # ),
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


if __name__ == '__main__':
    unittest.main()
