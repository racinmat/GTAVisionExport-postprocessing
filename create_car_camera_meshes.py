import scipy.io

from stl import mesh
import math
import numpy as np
# Optionally render the rotated cube faces
from matplotlib import pyplot
from mpl_toolkits import mplot3d
from math import sin, cos


def create_rot_matrix(rot):
    x = np.radians(rot[0])
    y = np.radians(rot[1])
    z = np.radians(rot[2])

    Rx = np.array([
        [1, 0, 0],
        [0, cos(x), -sin(x)],
        [0, sin(x), cos(x)]
    ], dtype=np.float)
    Ry = np.array([
        [cos(y), 0, sin(y)],
        [0, 1, 0],
        [-sin(y), 0, cos(y)]
    ], dtype=np.float)
    Rz = np.array([
        [cos(z), -sin(z), 0],
        [sin(z), cos(z), 0],
        [0, 0, 1]
    ], dtype=np.float)
    result = Rz @ Ry @ Rx
    return result


def create_base_cam_mesh():
    # Create tetrahedron
    data = np.zeros(6, dtype=mesh.Mesh.dtype)

    # this way, tetrahedron has base in y axis and peax in the middle of coords
    # tetrahedron base 1
    data['vectors'][0] = np.array(
        [[1, 2, 1],
         [1, 2, -1],
         [-1, 2, -1]])
    # tetrahedron base 2
    data['vectors'][1] = np.array(
        [[1, 2, 1],
         [-1, 2, 1],
         [-1, 2, -1]])
    # sides
    data['vectors'][2] = np.array(
        [[1, 2, 1],
         [-1, 2, 1],
         [0, 0, 0]])
    data['vectors'][3] = np.array(
        [[1, 2, 1],
         [1, 2, -1],
         [0, 0, 0]])
    data['vectors'][4] = np.array(
        [[-1, 2, -1],
         [-1, 2, 1],
         [0, 0, 0]])
    data['vectors'][5] = np.array(
        [[-1, 2, -1],
         [1, 2, -1],
         [0, 0, 0]])

    return data


def cuboid_to_trianges(xmin, ymin, zmin, xmax, ymax, zmax):
    return np.array([
        # xmin const
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
        ],
        [
            [xmin, ymax, zmax],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
        ],
        # xmax const
        [
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
        ],
        [
            [xmax, ymax, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
        ],
        # ymin const
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmin],
        ],
        [
            [xmax, ymin, zmax],
            [xmin, ymin, zmax],
            [xmax, ymin, zmin],
        ],
        # ymax const
        [
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymax, zmin],
        ],
        [
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmin],
        ],
        # zmin const
        [
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymin, zmin],
        ],
        [
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmax, ymin, zmin],
        ],
        # zmax const
        [
            [xmin, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymin, zmax],
        ],
        [
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
            [xmax, ymin, zmax],
        ],
    ])


def create_base_car_mesh():
    # Create shape
    data = np.zeros(12, dtype=mesh.Mesh.dtype)

    # the shape will be something like cuboid, but squeezed in the front
    # sizes from car bounding box, but modified
    data['vectors'] = cuboid_to_trianges(-0.91, -2.30, -0.57, 0.91, 2.14, 0.98)

    return data


def create_cam_mesh(position, rotation, rel_rotation):
    data = create_base_cam_mesh()
    for key, arr in enumerate(data['vectors']):
        # how rotation is now
        # data['vectors'][key] = data['vectors'][key] @ create_rot_matrix(rotation).T
        # how rotation should be
        data['vectors'][key] = data['vectors'][key] @ (create_rot_matrix(rotation) @ create_rot_matrix(rel_rotation)).T
        data['vectors'][key] += position
    m = mesh.Mesh(data.copy())
    return m


def create_car_mesh(position, rotation):
    data = create_base_car_mesh()
    for key, arr in enumerate(data['vectors']):
        data['vectors'][key] = data['vectors'][key] @ create_rot_matrix(rotation).T
        data['vectors'][key] += position
    m = mesh.Mesh(data.copy())
    return m


if __name__ == '__main__':
    cam_positions = (
        (1179.651, -2045.311, 46.579),
        (1178.465, -2044.489, 46.677),
        (1175.448, -2044.995, 45.730),
        (1178.449, -2046.015, 46.194),
    )
    cam_rotations = (
        (11.961, 17.577, -90.631),
        (11.961, 17.577, -0.631),
        (11.961, 17.577, 89.368),
        (11.961, 17.577, 179.368),
    )
    cam_rel_rotations = (
        (0, 0, 0),
        (0, 0, 90),
        (0, 0, 180),
        (0, 0, 270),
    )

    for i in range(len(cam_positions)):
        # how it is now
        # pos, rot, rel_rot = cam_positions[i], cam_rotations[i], cam_rel_rotations[i]
        # how it should be
        pos, rot, rel_rot = cam_positions[i], cam_rotations[0], cam_rel_rotations[i]    # 0th is car rotation
        m = create_cam_mesh(pos, rot, rel_rot)
        m.save('./example/camera-{}.stl'.format(i))

    m = create_car_mesh((1177.75, -2045.07, 45.90), (11.96167850494384800, 17.57795333862304700, -90.63151550292969000))
    m.save('./example/car.stl')
