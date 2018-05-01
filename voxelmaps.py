import warnings

import numpy as np
from gta_math import construct_view_matrix, construct_proj_matrix, points_to_homo, ndc_to_view, view_to_world, \
    ndcs_to_pixels, get_depth_lut_for_linear_view
from pointcloud_to_voxelmap import pointclouds_to_voxelmap, pointclouds_to_voxelmap_with_map
from visualization import get_connection_pooled, load_depth
import time


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
    # print('MAX_DISTANCE', MAX_DISTANCE)
    # start = time.time()
    name = cam['imagepath']
    depth = load_depth(name)
    cam['cam_far_clip'] = MAX_DISTANCE
    # print('load_depth:', time.time() - start)
    # start = time.time()
    vecs, _ = points_to_homo(cam, depth)
    # print('points_to_homo:', time.time() - start)
    # start = time.time()
    assert(vecs.shape[0] == 4)
    vecs_p = ndc_to_view(vecs, cam['proj_matrix'])
    vecs_p_world = view_to_world(vecs_p, cam['view_matrix'])
    # print('ndc_to_world:', time.time() - start)
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


def subsample_pointcloud(pointcloud, subsampling_size=1e-1):
    import pcl
    assert (type(subsampling_size) == float)
    p = pcl.PointCloud(pointcloud.astype(dtype=np.float32).T)
    pcl_voxelmap = p.make_voxel_grid_filter()
    pcl_voxelmap.set_leaf_size(x=subsampling_size, y=subsampling_size, z=subsampling_size)
    filtered_p = pcl_voxelmap.filter()
    filtered_pcl = filtered_p.to_array().T
    if filtered_pcl.shape == pointcloud.shape:
        warnings.warn("pointcloud is same size after subsampling, something is wrong, probably voxelsize too small")
    return filtered_pcl


def scene_to_pointcloud(cameras, subsampling_size=None):
    pointclouds = []
    cam_positions = []

    for cam in cameras:
        # start = time.time()
        pointcloud = camera_to_pointcloud(cam)
        # print('camera_to_pointcloud:', time.time() - start)
        # start = time.time()
        if subsampling_size is not None:
            # print('performing pcl subsampling with size {}'.format(subsampling_size))
            pointcloud = subsample_pointcloud(pointcloud, subsampling_size=subsampling_size)
        # print('subsample_pointcloud:', time.time() - start)
        pointclouds.append(pointcloud)
        cam_positions.append(cam['camera_pos'])
    return pointclouds, cam_positions


def scene_to_voxelmap(cameras):
    voxels, values, voxel_size, map_obj = scene_to_voxelmap_with_map(cameras)

    return voxels, values, voxel_size


def to_main_cam_view(cameras, pointclouds, cam_positions):
    main_cam = get_main_image(cameras)
    view_matrix = main_cam['view_matrix']
    for i, (pointcloud, cam_pos) in enumerate(zip(pointclouds, cam_positions)):
        pointcloud_view = view_matrix @ np.vstack([pointcloud, np.ones((1, pointcloud.shape[1]))])
        pointcloud_view /= pointcloud_view[3, :]
        pointclouds[i] = pointcloud_view[0:3, :]

        cam_pos_view = view_matrix @ np.hstack([cam_pos, 1])
        cam_pos_view /= cam_pos_view[3]
        cam_positions[i] = cam_pos_view[0:3]
    return pointclouds, cam_positions


def scene_to_voxelmap_with_map(cameras, subsampling_size=None, main_camera_view=False):
    # if main_camera_view parameter is true, coordinates will be transferred to main camera coordinate system
    # this method is just fucking slow, because of pointclouds_to_voxelmap_with_map
    # start = time.time()
    pointclouds, cam_positions = scene_to_pointcloud(cameras, subsampling_size)
    if main_camera_view:
        pointclouds, cam_positions = to_main_cam_view(cameras, pointclouds, cam_positions)

    # print('scene_to_pointcloud:', time.time() - start)
    # start = time.time()
    assert (pointclouds[0].shape[0] == 3)
    voxels, values, voxel_size, map_obj = pointclouds_to_voxelmap_with_map(pointclouds, cam_positions)
    # print('pointclouds_to_voxelmap_with_map:', time.time() - start)

    return voxels, values, voxel_size, map_obj


def ndc_pcl_to_grid_linear_view(x_range, y_range, z_range, occupied_ndc_positions, proj_matrix, z_meters_min, z_meters_max):
    bin_size = (z_meters_max - z_meters_min) / z_range
    ndc_z_min = get_depth_lut_for_linear_view(proj_matrix, z_meters_min + (bin_size / 2), z_meters_max + (bin_size / 2), z_range)
    return ndc_pcl_to_grid_with_lut(x_range, y_range, z_range, occupied_ndc_positions, ndc_z_min)


def ndc_pcl_to_grid_with_lut(x_range, y_range, z_range, occupied_ndc_positions, ndc_z_min):
    # here bin borders are in lut
    # because I want to have Z scale linear in view, and nonlinear in NDC, I need to perform nonlinear binning of z values

    # now I create x X y X z grid with 0s and 1s as grid
    # so now I have data in pointcloud. And I need to convert these NDC values
    # into indices, so x:[-1, 1] into [0, 239], y:[-1, 1] to [0, 159],
    # and z:[z_min, z_max] into [0, 99]
    voxelmap_ndc_grid = np.zeros((x_range, y_range, z_range), dtype=np.int8)
    vecs = ndcs_to_pixels(occupied_ndc_positions[0:2, :], (y_range, x_range))
    vec_y = vecs[0, :]
    vec_x = vecs[1, :]
    vec_z = np.digitize(occupied_ndc_positions[2, :], ndc_z_min)
    vec_z[vec_z >= z_range] = z_range - 1   # just throw outliers into nearest bin
    voxelmap_ndc_grid[vec_x, vec_y, vec_z] = 1
    return voxelmap_ndc_grid


def generate_frustum_points(proj_matrix, x_range, y_range, z_range, z_meters_min, z_meters_max, linear_view_sampling):
    x_min = -1  # NDC corners
    x_max = 1  # NDC corners
    y_min = -1  # NDC corners
    y_max = 1  # NDC corners
    # z min calc
    z_min = proj_matrix @ [1, 1, -z_meters_max, 1]
    z_min = z_min[2] / z_min[3]
    # z max calc
    z_max = proj_matrix @ [1, 1, -z_meters_min, 1]
    z_max = z_max[2] / z_max[3]

    if proj_matrix.tobytes() in points_cache:
        return points_cache[proj_matrix.tobytes()].copy(), z_max, z_min

    X, Y, Z, W = np.meshgrid(np.linspace(x_min, x_max, x_range), np.linspace(y_min, y_max, y_range),
                             np.linspace(z_min, z_max, z_range), np.linspace(1, 2, 1))
    if linear_view_sampling:
        X_view, Y_view, Z_view, W_view = np.meshgrid(np.linspace(1, 2, 1), np.linspace(1, 2, 1),
                                                     np.linspace(-z_meters_max, -z_meters_min, z_range),
                                                     np.linspace(1, 2, 1))
        view_positions = np.vstack([X_view.ravel(), Y_view.ravel(), Z_view.ravel(), W_view.ravel()])
        ndc_positions = proj_matrix @ view_positions
        ndc_positions /= ndc_positions[3, :]
        ndc_z = ndc_positions[2, :]
        Z = np.tile(ndc_z, (x_range, y_range, 1))[:, :, :, np.newaxis]

    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), W.ravel()])

    points_cache[proj_matrix.tobytes()] = positions
    return positions, z_max, z_min


points_cache = {}
