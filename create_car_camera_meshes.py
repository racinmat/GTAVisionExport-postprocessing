import scipy.io

from stl import mesh
import math
import numpy as np
# Optionally render the rotated cube faces
from matplotlib import pyplot
from mpl_toolkits import mplot3d
from math import sin, cos

from visualization import get_connection


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
        data['vectors'][key] = data['vectors'][key] @ create_rot_matrix(rotation).T
        # how rotation should be
        # data['vectors'][key] = data['vectors'][key] @ (create_rot_matrix(rotation) @ create_rot_matrix(rel_rotation)).T
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


def load_data_db_scene(scene_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""SELECT 
        ARRAY[st_x(camera_pos), st_y(camera_pos), st_z(camera_pos)] AS camera_pos,
        ARRAY[st_x(camera_rot), st_y(camera_rot), st_z(camera_rot)] AS camera_rot,
        ARRAY[st_x(player_pos), st_y(player_pos), st_z(player_pos)] AS player_pos
        FROM snapshots
        WHERE scene_id = '{}'
        ORDER BY snapshot_id ASC
    """.format(scene_id))
    # print(size)
    cam_rel_rotations = (
        (0, 0, 0),
        (0, 0, 90),
        (0, 0, 180),
        (0, 0, 270),
    )
    results = []
    for row in cur:
        res = dict(row)
        results.append(res)
    cam_positions = [i['camera_pos'] for i in results]
    cam_rotations = [i['camera_rot'] for i in results]
    car_position = results[0]['player_pos']
    return cam_positions, cam_rotations, cam_rel_rotations, car_position


def load_data_static():
    # cam_positions = (
    #     (1179.651, -2045.311, 46.579),
    #     (1178.465, -2044.489, 46.677),
    #     (1175.448, -2044.995, 45.730),
    #     (1178.449, -2046.015, 46.194),
    # )
    # cam_rotations = (
    #     (11.961, 17.577, -90.631),
    #     (11.961, 17.577, -0.631),
    #     (11.961, 17.577, 89.368),
    #     (11.961, 17.577, 179.368),
    # )
    # car_position = (1177.75, -2045.07, 45.90)
    cam_positions = (
        (1178.89111328125000000, -2040.52697753906250000, 47.88730621337890600),
        (1177.46374511718750000, -2040.32214355468750000, 47.78350830078125000),
        (1175.13684082031250000, -2042.16235351562500000, 46.57566070556640600),
        (1178.18627929687500000, -2041.70898437500000000, 47.44472503662109400),
    )
    cam_rotations = (
        (18.18645095825195300, 12.22425079345703100, -62.48254776000976600),
        (18.18645095825195300, 12.22425365447998000, 27.51745223999023400),
        (18.18644905090332000, 12.22425365447998000, 117.51745605468750000),
        (18.18644905090332000, 12.22425556182861300, -152.48252868652344000),
    )
    car_position = (1177.20007324218750000, -2041.19079589843750000, 46.99868774414062500)
    cam_rel_rotations = (
        (0, 0, 0),
        (0, 0, 90),
        (0, 0, 180),
        (0, 0, 270),
    )
    return cam_positions, cam_rotations, cam_rel_rotations, car_position


if __name__ == '__main__':
    # cam_positions, cam_rotations, cam_rel_rotations, car_position = load_data_static()
    # cam_positions, cam_rotations, cam_rel_rotations, car_position = load_data_db_scene('412c1831-419a-4459-83d2-9a6eaf89a62f')
    cam_positions, cam_rotations, cam_rel_rotations, car_position = load_data_db_scene('1ce01ac0-f41e-48f0-a78f-867575a50aa4')
    for i in range(len(cam_positions)):
        # how it is now
        pos, rot, rel_rot = cam_positions[i], cam_rotations[i], cam_rel_rotations[i]
        # how it should be
        # pos, rot, rel_rot = cam_positions[i], cam_rotations[0], cam_rel_rotations[i]  # 0th is car rotation
        m = create_cam_mesh(pos, rot, rel_rot)
        m.save('./example/camera-{}.stl'.format(i))

    m = create_car_mesh(car_position, cam_rotations[0])
    m.save('./example/car.stl')
