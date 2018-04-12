import numpy as np
from gta_math import construct_view_matrix, construct_proj_matrix, points_to_homo, ndc_to_view, view_to_world
from pointcloud_to_voxelmap import pointclouds_to_voxelmap
from visualization import get_connection_pooled, load_depth

MAX_DISTANCE = 20


class NoMainImageException(Exception):
    pass


def load_scene_db_data(scene_id):
    conn = get_connection_pooled()
    cur = conn.cursor()
    cur.execute("""SELECT snapshot_id, imagepath, cam_near_clip, camera_fov, width, height, \
      ARRAY[st_x(camera_relative_rotation), st_y(camera_relative_rotation), st_z(camera_relative_rotation)] as camera_relative_rotation, \
      ARRAY[st_x(camera_pos), st_y(camera_pos), st_z(camera_pos)] as camera_pos, \
      ARRAY[st_x(camera_rot), st_y(camera_rot), st_z(camera_rot)] as camera_rot \
      FROM snapshots \
      WHERE scene_id = '{}'
      ORDER BY timestamp ASC \
    """.format(scene_id))

    cameras = []
    for row in cur:
        res = dict(row)
        res['camera_rot'] = np.array(res['camera_rot'])
        res['camera_pos'] = np.array(res['camera_pos'])
        res['camera_relative_rotation'] = np.array(res['camera_relative_rotation'])
        res['view_matrix'] = construct_view_matrix(res['camera_pos'], res['camera_rot'])
        res['proj_matrix'] = construct_proj_matrix(res['height'], res['width'], res['camera_fov'], res['cam_near_clip'])
        cameras.append(res)
    return cameras


def camera_to_pointcloud(cam):
    name = cam['imagepath']
    depth = load_depth(name)
    cam['cam_far_clip'] = MAX_DISTANCE
    vecs, _ = points_to_homo(cam, depth)
    assert(vecs.shape[0] == 4)
    vecs_p = ndc_to_view(vecs, cam['proj_matrix'])
    vecs_p_world = view_to_world(vecs_p, cam['view_matrix'])
    assert(vecs_p_world.shape[0] == 4)
    return vecs_p_world[0:3, :]


def get_main_image_name(cameras):
    for cam in cameras:
        # this is the main camera
        if np.array_equal(cam['camera_relative_rotation'], [0, 0, 0]):
            return cam['imagepath']
    raise NoMainImageException('no main image')


def get_main_image(cameras):
    for cam in cameras:
        # this is the main camera
        if np.array_equal(cam['camera_relative_rotation'], [0, 0, 0]):
            return cam
    raise NoMainImageException('no main image')


def scene_to_pointcloud(cameras):
    pointclouds = []
    cam_positions = []

    for cam in cameras:
        pointcloud = camera_to_pointcloud(cam)
        pointclouds.append(pointcloud)
        cam_positions.append(cam['camera_pos'])
    return pointclouds, cam_positions


def scene_to_voxelmap(scene_id):
    # start = time.time()
    pointclouds, cam_positions = scene_to_pointcloud(scene_id)
    # end = time.time()
    # print('scene to pointclouds', end - start)
    # start = time.time()
    assert (pointclouds[0].shape[0] == 3)
    voxelmap = pointclouds_to_voxelmap(pointclouds, cam_positions)
    # end = time.time()
    # print('pointclouds to voxelmap', end - start)

    return voxelmap
