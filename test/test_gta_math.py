import unittest
import numpy as np

from gta_math import construct_model_matrix, create_model_rot_matrix, rot_matrix_to_euler_angles, create_rot_matrix, \
    model_rot_matrix_to_euler_angles, car_and_relative_cam_to_absolute_cam_rotation_matrix_1, \
    car_and_relative_cam_to_absolute_cam_rotation_matrix
from toyota import get_my_car_position_and_rotation


class TestGtaMath(unittest.TestCase):

    def test_model_euler_angles_matrix_transformation_1(self):
        orig_angles = np.array([0, 0, 0])
        matrix = create_model_rot_matrix(orig_angles)
        angles = model_rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_model_euler_angles_matrix_transformation_2(self):
        orig_angles = np.array([0, 90, 0])
        matrix = create_model_rot_matrix(orig_angles)
        angles = model_rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_model_euler_angles_matrix_transformation_3(self):
        orig_angles = np.array([0, 0, 90])
        matrix = create_model_rot_matrix(orig_angles)
        angles = model_rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_model_euler_angles_matrix_transformation_4(self):
        orig_angles = np.array([90, 0, 0])
        matrix = create_model_rot_matrix(orig_angles)
        angles = model_rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_model_euler_angles_matrix_transformation_5(self):
        orig_angles = np.array([40, 50, 60])
        matrix = create_model_rot_matrix(orig_angles)
        angles = model_rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_model_euler_angles_matrix_transformation_6(self):
        orig_angles = np.array([20, 30, 0])
        matrix = create_model_rot_matrix(orig_angles)
        angles = model_rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_euler_angles_matrix_transformation_1(self):
        orig_angles = np.array([0, 0, 0])
        matrix = create_rot_matrix(orig_angles)
        angles = rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_euler_angles_matrix_transformation_2(self):
        orig_angles = np.array([0, 90, 0])
        matrix = create_rot_matrix(orig_angles)
        angles = rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_euler_angles_matrix_transformation_3(self):
        orig_angles = np.array([0, 0, 90])
        matrix = create_rot_matrix(orig_angles)
        angles = rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_euler_angles_matrix_transformation_4(self):
        orig_angles = np.array([90, 0, 0])
        matrix = create_rot_matrix(orig_angles)
        angles = rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_euler_angles_matrix_transformation_5(self):
        orig_angles = np.array([40, 50, 60])
        matrix = create_rot_matrix(orig_angles)
        angles = rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_euler_angles_matrix_transformation_6(self):
        orig_angles = np.array([20, 30, 0])
        matrix = create_rot_matrix(orig_angles)
        angles = rot_matrix_to_euler_angles(matrix)
        self.assertTrue(np.allclose(angles, orig_angles))

    def test_car_to_camera_view_rotation_1(self):
        correct_car_rot = np.array([-0.006126282270997763, 0.84723699092864990, 0.44408929347991943])
        cam_rot = np.array([0.84683728218078610, 0.007344349753111601, 90.41197204589844000])
        cam_rel_rot = np.array([0., 0., 90.])
        orig_cam_rot_m = create_rot_matrix(cam_rot)
        cam_rot_m = car_and_relative_cam_to_absolute_cam_rotation_matrix(correct_car_rot, cam_rel_rot)
        self.assertTrue(np.allclose(cam_rot_m, orig_cam_rot_m, atol=3.e-3))

    def test_car_to_camera_view_rotation_2(self):
        correct_car_rot = np.array([-0.006126282270997763, 0.84723699092864990, 0.44408929347991943])
        cam_rot = np.array([0.008536399342119694, -0.84658384323120120, -179.62216186523438000])
        cam_rel_rot = np.array([0., 0., 180.])
        orig_cam_rot_m = create_rot_matrix(cam_rot)
        cam_rot_m = car_and_relative_cam_to_absolute_cam_rotation_matrix(correct_car_rot, cam_rel_rot)
        self.assertTrue(np.allclose(cam_rot_m, orig_cam_rot_m, atol=3.e-3))

    def test_car_to_camera_view_rotation_3(self):
        correct_car_rot = np.array([-0.006126282270997763, 0.84723699092864990, 0.44408929347991943])
        cam_rot = np.array([-0.84574091434478760, -0.011379104107618332, -89.68127441406250000])
        cam_rel_rot = np.array([0., 0., 270.])
        orig_cam_rot_m = create_rot_matrix(cam_rot)
        cam_rot_m = car_and_relative_cam_to_absolute_cam_rotation_matrix(correct_car_rot, cam_rel_rot)
        self.assertTrue(np.allclose(cam_rot_m, orig_cam_rot_m, atol=3.e-3))


if __name__ == '__main__':
    unittest.main()
