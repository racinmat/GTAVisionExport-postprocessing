import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
import json
from matplotlib import patches, cm
from math import tan, atan, radians, degrees, cos, sin
from mpl_toolkits.mplot3d import Axes3D
import glob
import progressbar
from joblib import Parallel, delayed
from display_voxelmap import save_csv
from gta_math import *
from visualization import save_pointcloud_csv, load_depth
from voxelmaps import ndc_pcl_to_grid_linear_view, camera_to_pointcloud, load_scene_db_data, scene_to_pointcloud
import visualization
from configparser import ConfigParser
import voxelmaps
import gta_math
import time
from moviepy.editor import *


def get_base_name(name):
    return os.path.basename(os.path.splitext(name)[0])


def check_plane(X0, X1, X2, x, y, z):
    v1 = X1 - X0
    v2 = X2 - X0
    n = np.cross(v1, v2)

    return n[0] * (x - X0[0]) + n[1] * (y - X0[1]) + n[2] * (z - X0[2]) > 0


def draw3dbboxes(directory, base_name):
    rgb_file = '{}/{}.jpg'.format(directory, base_name)
    json_file = '{}/{}.json'.format(directory, base_name)
    depth_file = '{}/{}-depth.png'.format(directory, base_name)
    stencil_file = '{}/{}-stencil.png'.format(directory, base_name)

    rgb = np.array(Image.open(rgb_file))
    depth = np.array(Image.open(depth_file))
    depth = depth / np.iinfo(np.uint16).max  # normalizing into NDC
    stencil = np.array(Image.open(stencil_file))
    with open(json_file, mode='r') as f:
        data = json.load(f)
    entities = data['entities']
    view_matrix = np.array(data['view_matrix'])
    proj_matrix = np.array(data['proj_matrix'])
    width = data['width']
    height = data['height']
    # visible_cars = [e for e in entities if e['bbox'][0] != [np.inf, np.inf] and e['type'] == 'car']
    visible_cars = [e for e in entities if
                    e['type'] == 'car' and e['class'] != 'Trains' and is_entity_in_image(depth, stencil, e, view_matrix,
                                                                                         proj_matrix, width, height)]

    print('camera pos: ', data['camera_pos'])
    fig = plt.figure(figsize=(16, 10))
    plt.axis('off')
    #    plt.xlim([0, rgb.shape[1]])
    #    plt.ylim([rgb.shape[0], 0])
    ax = plt.gca()
    plt.imshow(rgb)

    for row in visible_cars:
        row['bbox_calc'] = calculate_2d_bbox(row['pos'], row['rot'], row['model_sizes'], view_matrix, proj_matrix, width, height)
        is_entity_in_image(depth, stencil, row, view_matrix, proj_matrix, width, height)
        # position in world coords
        # print(row)
        pos = np.array(row['pos'])
        pixel_pos = world_coords_to_pixel(pos, view_matrix, proj_matrix, width, height)
        # print('entity pos in image:', pos)
        # print('entity pos in image:', pixel_pos)

        bbox = np.array(row['bbox_calc'])
        bbox[:, 0] *= width
        bbox[:, 1] *= height
        bbox_width, bbox_height = bbox[0, :] - bbox[1, :]
        print('2D bbox:', bbox)
        rect = patches.Rectangle(bbox[1, :], bbox_width, bbox_height, linewidth=1, edgecolor='y', facecolor='none')
        ax.add_patch(rect)

        # 3D bounding box
        rot = np.array(row['rot'])
        model_sizes = np.array(row['model_sizes'])
        points_3dbbox = get_model_3dbbox(model_sizes)

        # projecting cuboid to 2d
        bbox_2d = model_coords_to_pixel(pos, rot, points_3dbbox, view_matrix, proj_matrix, width, height).T
        # print('3D bbox:\n', points_3dbbox)
        # print('3D bbox in 2D:\n', bbox_2d)

        # showing cuboid
        pol1 = patches.Polygon(bbox_2d[(0, 1, 3, 2), :], closed=True, linewidth=1, edgecolor='c',
                               facecolor='none')  # fixed x
        pol2 = patches.Polygon(bbox_2d[(4, 5, 7, 6), :], closed=True, linewidth=1, edgecolor='c',
                               facecolor='none')  # fixed x
        pol3 = patches.Polygon(bbox_2d[(0, 2, 6, 4), :], closed=True, linewidth=1, edgecolor='c',
                               facecolor='none')  # fixed z
        pol4 = patches.Polygon(bbox_2d[(1, 3, 7, 5), :], closed=True, linewidth=1, edgecolor='c',
                               facecolor='none')  # fixed z
        pol5 = patches.Polygon(bbox_2d[(0, 1, 5, 4), :], closed=True, linewidth=1, edgecolor='r',
                               facecolor='none')  # fixed y
        pol6 = patches.Polygon(bbox_2d[(2, 3, 7, 6), :], closed=True, linewidth=1, edgecolor='g',
                               facecolor='none')  # fixed y
        ax.add_patch(pol1)
        ax.add_patch(pol2)
        ax.add_patch(pol3)
        ax.add_patch(pol4)
        ax.add_patch(pol5)
        ax.add_patch(pol6)

        # just some dumping for debug purposes
        point_homo = np.array(
            [points_3dbbox[:, 0], points_3dbbox[:, 1], points_3dbbox[:, 2], np.ones_like(points_3dbbox[:, 0])])
        model_matrix = construct_model_matrix(pos, rot)
        point_homo = model_matrix @ point_homo
        viewed = view_matrix @ point_homo
        projected = proj_matrix @ viewed
        # projected[0:3, projected[ 3] < 0] *= -1
        projected /= projected[3]
        np.savetxt("../sample-images/projected-2.csv", projected.T, delimiter=",")
        np.savetxt("../sample-images/viewed-2.csv", viewed.T, delimiter=",")
    plt.show()


def draw_car_pixels(in_directory, out_directory, base_name):
    rgb_file = '{}/{}.jpg'.format(in_directory, base_name)
    json_file = '{}/{}.json'.format(in_directory, base_name)
    depth_file = '{}/{}-depth.png'.format(in_directory, base_name)
    stencil_file = '{}/{}-stencil.png'.format(in_directory, base_name)

    rgb = np.array(Image.open(rgb_file))
    depth = np.array(Image.open(depth_file))
    depth = depth / np.iinfo(np.uint16).max  # normalizing into NDC
    stencil = np.array(Image.open(stencil_file))
    with open(json_file, mode='r') as f:
        data = json.load(f)
    entities = data['entities']
    view_matrix = np.array(data['view_matrix'])
    proj_matrix = np.array(data['proj_matrix'])
    width = data['width']
    height = data['height']
    # visible_cars = [e for e in entities if e['bbox'][0] != [np.inf, np.inf] and e['type'] == 'car']
    visible_cars = [e for e in entities if
                    e['class'] != 'Trains' and is_entity_in_image(depth, stencil, e, view_matrix, proj_matrix, width, height)]

    params = {
        'width': width,
        'height': height,
        'proj_matrix': proj_matrix,
        'cam_far_clip': 100,
    }
    pts, _ = points_to_homo(params, depth, False)  # False to get all pixels
    pts_p = ndc_to_view(pts, proj_matrix)
    pixel_3d = view_to_world(pts_p, view_matrix)
    pixel_3d = np.reshape(pixel_3d, (4, height, width))
    pixel_3d = pixel_3d[0:3, ::]

    cc, rr = np.meshgrid(range(width), range(height))
    car_mask = np.bitwise_and(stencil, 7) == 2

    plt.figure(figsize=(16, 10))
    plt.axis('off')
    plt.xlim([0, rgb.shape[1]])
    plt.ylim([rgb.shape[0], 0])

    id = 1
    for row in visible_cars:
        row['bbox_calc'] = calculate_2d_bbox(row['pos'], row['rot'], row['model_sizes'], view_matrix, proj_matrix, width, height)
        # position in world coords
        pos = np.array(row['pos'])
        rot = np.array(row['rot'])

        model_sizes = np.array(row['model_sizes'])
        points_3dbbox = get_model_3dbbox(model_sizes)

        # 3D bounding box
        # projecting cuboid to 2d
        bbox_3d = model_coords_to_world(pos, rot, points_3dbbox, view_matrix, proj_matrix)
        # for i, point in enumerate(points_3dbbox):
        #     # point += pos
        #     bbox_3d[i, :] = model_coords_to_world(pos, rot, point, view_matrix, proj_matrix, width, height)

        # bounding box
        bbox = np.array(row['bbox_calc'])
        # bbox = np.array(row['bbox'])
        bbox[:, 0] *= width
        bbox[:, 1] *= height
        bbox_width, bbox_height = bbox[0, :] - bbox[1, :]

        bbox = np.array([[np.ceil(bbox[0, 0]), np.floor(bbox[0, 1])],
                         [np.ceil(bbox[1, 0]), np.floor(bbox[1, 1])]]).astype(int)

        # points inside the 2D bbox with car mask on
        idxs = np.where(
            (car_mask == True) & (cc >= bbox[1, 0]) & (cc <= bbox[0, 0]) & (rr >= bbox[1, 1]) & (rr <= bbox[0, 1]))

        # zkusit na 25 metrů, a mít lineární hloubky jako baseline
        # 3D coordinates of pixels in idxs
        x = pixel_3d[0, ::].squeeze()[idxs]
        y = pixel_3d[1, ::].squeeze()[idxs]
        z = pixel_3d[2, ::].squeeze()[idxs]

        # test if the points lie inside 3D bbox
        in1 = check_plane(bbox_3d[3, :], bbox_3d[2, :], bbox_3d[7, :], x, y, z)
        in2 = check_plane(bbox_3d[1, :], bbox_3d[5, :], bbox_3d[0, :], x, y, z)
        in3 = check_plane(bbox_3d[6, :], bbox_3d[2, :], bbox_3d[4, :], x, y, z)
        in4 = check_plane(bbox_3d[3, :], bbox_3d[7, :], bbox_3d[1, :], x, y, z)
        in5 = check_plane(bbox_3d[7, :], bbox_3d[6, :], bbox_3d[5, :], x, y, z)
        in6 = check_plane(bbox_3d[0, :], bbox_3d[2, :], bbox_3d[1, :], x, y, z)
        is_inside = in1 & in2 & in3 & in4 & in5 & in6

        rgb[(idxs[0][is_inside], idxs[1][is_inside])] = np.array(cm.viridis(id * 25)[0:3]) * 255

        id += 1
        # assuming there are no more than 10 cars in image
    #    plt.imshow(car_id_mask)
    #    plt.clim(0.1, 1)
    plt.imshow(rgb)
    # plt.savefig('{}/{}.jpg'.format(out_directory, base_name), bbox_inches='tight')
    plt.show()


def try_points_to_grid_and_back():
    proj_matrix = np.array([[1.21006660e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00],
                            [0.00000000e+00, 2.14450692e+00, 0.00000000e+00,
                             0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00, 1.49965283e-04,
                             1.50022495e+00],
                            [0.00000000e+00, 0.00000000e+00, -1.00000000e+00,
                             0.00000000e+00]])

    # data preparation
    # x_range = 160
    # y_range = 120
    # z_range = 100
    x_range = 8
    y_range = 6
    z_range = 5
    z_meters_min = 1.5
    z_meters_max = 25
    # z min calc
    z_min = proj_matrix @ [1, 1, -z_meters_max, 1]
    z_min = z_min[2] / z_min[3]
    # z max calc
    z_max = proj_matrix @ [1, 1, -z_meters_min, 1]
    z_max = z_max[2] / z_max[3]

    ndc_points = np.array([
        [-1, 1, 0.6, 1],
        [-0.5, 0.4, 0.6, 1],
        [1, -0.8, 0.4, 1],
        [-0.4, -1, 0.4, 1],
        [0.4, 0.6, 0.8, 1],
    ]).T

    view_points = ndc_to_view(ndc_points, proj_matrix)
    ndc_grid = ndc_pcl_to_grid_linear_view(x_range, y_range, z_range, ndc_points, proj_matrix, z_meters_min, z_meters_max)
    ndc_points_reconst = grid_to_ndc_pcl_linear_view(ndc_grid, proj_matrix, z_meters_min, z_meters_max)
    ndc_points_reconst = np.hstack((ndc_points_reconst, np.ones((ndc_points_reconst.shape[0], 1)))).T

    view_points_reconst = ndc_to_view(ndc_points_reconst, proj_matrix)

    save_pointcloud_csv(ndc_points.T[:, 0:3], '{}/ndc-{}.csv'.format('../sample-images', 'sample'))
    save_pointcloud_csv(view_points.T[:, 0:3], '{}/view-{}.csv'.format('../sample-images', 'sample'))
    save_pointcloud_csv(ndc_points_reconst.T[:, 0:3], '{}/ndc-reconst-{}.csv'.format('../sample-images', 'sample'))
    save_pointcloud_csv(view_points_reconst.T[:, 0:3], '{}/view-reconst-{}.csv'.format('../sample-images', 'sample'))

    bin_size = (z_meters_max - z_meters_min) / z_range
    print('z view voxel size: {}'.format(bin_size))
    ndc_z = get_depth_lut_for_linear_view(proj_matrix, z_meters_min, z_meters_max, z_range)
    ndc_z_min = get_depth_lut_for_linear_view(proj_matrix, z_meters_min + (bin_size / 2), z_meters_max + (bin_size / 2), z_range)
    # now I calculate min value of each bin for binning values
    # ndc_z_min = get_depth_lut_for_linear_view(proj_matrix, z_meters_min + (bin_size / 2), z_meters_max + (bin_size / 2), z_range)
    plt.figure(figsize=(20, 7))

    # just playing with transformations here
    x_orig = ndc_points[0, :]
    vecs = ndcs_to_pixels(ndc_points[0:2, :], (y_range, x_range))
    x_vec = vecs[1, :]
    x_reconst = (x_vec / (0.5*x_range)) - 1

    plt.plot(ndc_points[0, :], x_vec, 'o', c='b')
    plt.plot(ndc_points_reconst[0, :], x_vec, 'o', c='r')
    # plt.plot(ndc_z, np.zeros_like(ndc_z), 'o', c='b')
    # plt.plot(ndc_z_min, np.zeros_like(ndc_z_min) - 0.1, 'o', c='r')
    # some_points = np.linspace(0, 1, 30)
    # digitized = np.digitize(some_points, ndc_z_min)
    # plt.plot(some_points, digitized, 'o', c='y')
    # plt.ylim([-1, 6])
    plt.show()


def try_subsampling():
    ini_file = "gta-postprocessing.ini"
    visualization.multi_page = False
    visualization.ini_file = ini_file

    conn = visualization.get_connection_pooled()
    cur = conn.cursor()

    CONFIG = ConfigParser()
    CONFIG.read(ini_file)
    in_directory = CONFIG["Images"]["Tiff"]
    out_directory = CONFIG["Images"]["MlDatasetVoxel"]
    out_inspect_directory = r'D:\showing-pointclouds'

    scene_id = '386b407b-586c-4d88-9d41-8dc2a0b70e70'  # from voxelmap run

    cameras = load_scene_db_data(scene_id)

    z_meters_min = 1.5
    z_meters_max = 25
    # voxelmaps.MAX_DISTANCE = 25
    voxelmaps.MAX_DISTANCE = 1500
    linear_view_sampling = True
    gta_math.PROJECTING = False

    start = time.time()
    pointclouds, cam_positions = scene_to_pointcloud(cameras, 1e-1)
    voxelmaps.scene_to_voxelmap_with_map(cameras, 1e-1, True)


def try_pcl_subsampling_detailed():
    import pcl
    ini_file = "gta-postprocessing.ini"
    visualization.multi_page = False
    visualization.ini_file = ini_file
    CONFIG = ConfigParser()
    CONFIG.read(ini_file)
    scene_id = '386b407b-586c-4d88-9d41-8dc2a0b70e70'  # from voxelmap run
    cameras = load_scene_db_data(scene_id)
    cam = cameras[0]

    z_meters_min = 1.5
    z_meters_max = 25
    voxelmaps.MAX_DISTANCE = 100
    linear_view_sampling = True
    # gta_math.PROJECTING = False
    gta_math.PROJECTING = True
    subsampling_size = 1e-1

    # getting pointcloud
    name = cam['imagepath']
    depth = visualization.load_depth(name)
    cam['cam_far_clip'] = voxelmaps.MAX_DISTANCE
    vecs, _ = points_to_homo(cam, depth)
    assert(vecs.shape[0] == 4)
    vecs_p = ndc_to_view(vecs, cam['proj_matrix'])
    vecs_p_world = view_to_world(vecs_p, cam['view_matrix'])
    assert(vecs_p_world.shape[0] == 4)
    pointcloud = vecs_p_world[0:3, :]
    print('pointcloud.shape[1]:', pointcloud.shape[1])

    # subsampling it with pcl
    p = pcl.PointCloud(pointcloud.astype(dtype=np.float32).T)
    pcl_voxelmap = p.make_voxel_grid_filter()
    pcl_voxelmap.set_leaf_size(x=subsampling_size, y=subsampling_size, z=subsampling_size)
    filtered_p = pcl_voxelmap.filter()
    pointcloud_subsampled = filtered_p.to_array().T

    print('pointcloud_subsampled.shape[1]:', pointcloud_subsampled.shape[1])

    name = cam['imagepath']
    depth = visualization.load_depth(name)
    cam['cam_far_clip'] = voxelmaps.MAX_DISTANCE
    vecs, _ = points_to_homo(cam, depth, False)
    assert(vecs.shape[0] == 4)
    vecs_p = ndc_to_view(vecs, cam['proj_matrix'])
    vecs_p_world = view_to_world(vecs_p, cam['view_matrix'])
    assert(vecs_p_world.shape[0] == 4)
    pointcloud = vecs_p_world[0:3, :]
    print('pointcloud.shape[1]:', pointcloud.shape[1])

    # subsampling it with pcl
    p = pcl.PointCloud(pointcloud.astype(dtype=np.float32).T)
    pcl_voxelmap = p.make_voxel_grid_filter()
    pcl_voxelmap.set_leaf_size(x=subsampling_size, y=subsampling_size, z=subsampling_size)
    filtered_p = pcl_voxelmap.filter()
    pointcloud_subsampled = filtered_p.to_array().T

    print('pointcloud_subsampled.shape[1]:', pointcloud_subsampled.shape[1])


def try_simple_pointcloud_load_and_merge():
    main_name = '2018-05-08--14-15-35--576'

    ini_file = "gta-postprocessing.ini"
    visualization.multi_page = False
    visualization.ini_file = ini_file

    conn = visualization.get_connection()
    cur = conn.cursor()

    cur.execute("""SELECT scene_id \
        FROM snapshots \
        WHERE imagepath = '{}' \
        """.format(main_name))

    scene_id = list(cur)[0]['scene_id']
    print(scene_id)

    cameras = load_scene_db_data(scene_id)

    voxelmaps.MAX_DISTANCE = 1000
    gta_math.PROJECTING = True

    #print([c['cam_far_clip'] for c in cameras])
    big_pcls, cam_positions = scene_to_pointcloud(cameras)
    print([c['cam_far_clip'] for c in cameras])
    # big_pcls, cam_positions = to_main_cam_view(cameras, big_pcls, cam_positions)
    big_pcl = np.hstack(big_pcls)
    print(big_pcl.max(axis=1) - big_pcl.min(axis=1))

    # save_pointcloud_csv(big_pcl.T[:, 0:3], '{}/big-orig-poincloud-{}.csv'.format(out_directory, main_name))


def try_one_car_3dbboxes(directory, base_name):
    rgb_file = '{}/{}.jpg'.format(directory, base_name)
    json_file = '{}/{}.json'.format(directory, base_name)
    depth_file = '{}/{}-depth.png'.format(directory, base_name)
    stencil_file = '{}/{}-stencil.png'.format(directory, base_name)

    rgb = np.array(Image.open(rgb_file))
    depth = np.array(Image.open(depth_file))
    depth = depth / np.iinfo(np.uint16).max  # normalizing into NDC
    stencil = np.array(Image.open(stencil_file))
    with open(json_file, mode='r') as f:
        data = json.load(f)
    entities = data['entities']
    view_matrix = np.array(data['view_matrix'])
    proj_matrix = np.array(data['proj_matrix'])
    width = data['width']
    height = data['height']
    # visible_cars = [e for e in entities if e['bbox'][0] != [np.inf, np.inf] and e['type'] == 'car']
    visible_cars = [e for e in entities if
                    e['class'] != 'Trains' and is_entity_in_image(depth, stencil, e, view_matrix, proj_matrix, width, height)]

    calculate_2d_bbox(visible_cars[1]['pos'], visible_cars[1]['rot'], visible_cars[1]['model_sizes'], view_matrix, proj_matrix, width, height)


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


def create_rot_matrix_2(rot):
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
    result = Rx @ Ry @ Rz
    return result


def create_rot_matrix_3(rot):
    x = np.radians(rot[0])
    y = np.radians(rot[1])
    z = np.radians(rot[2])

    Rx = np.array([
        [1, 0, 0],
        [0, sin(x), cos(x)],
        [0, cos(x), -sin(x)]
    ], dtype=np.float)
    Ry = np.array([
        [cos(y), 0, -sin(y)],
        [0, 1, 0],
        [sin(y), 0, cos(y)]
    ], dtype=np.float)
    Rz = np.array([
        [cos(z), sin(z), 0],
        [sin(z), -cos(z), 0],
        [0, 0, 1]
    ], dtype=np.float)
    result = Rx @ Ry @ Rz
    return result


def create_rot_matrix_4(rot):
    x = np.radians(rot[0])
    y = np.radians(rot[1])
    z = np.radians(rot[2])

    Rx = np.array([
        [1, 0, 0],
        [0, sin(x), cos(x)],
        [0, cos(x), -sin(x)]
    ], dtype=np.float)
    Ry = np.array([
        [cos(y), 0, -sin(y)],
        [0, 1, 0],
        [sin(y), 0, cos(y)]
    ], dtype=np.float)
    Rz = np.array([
        [cos(z), sin(z), 0],
        [sin(z), -cos(z), 0],
        [0, 0, 1]
    ], dtype=np.float)
    result = Rz @ Ry @ Rx
    return result


def play_with_matrices():
    cam_1_rot = [20.94379806518554700, 0.59126496315002440, 49.62200546264648400]
    cam_2_rot = [0.63308995962142940, -20.94262886047363300, 139.39569091796875000]
    cam_3_rot = [-20.94379806518554700, -0.59126579761505130, -130.37797546386720000]
    cam_4_rot = [-0.63308995962142940, 20.94263076782226600, -40.60428619384765600]

    car_rot = [20.94379806518554700, 0.59126496315002440, 49.62200546264648400]

    cam_1_rel_rot = [0, 0, 0]
    cam_2_rel_rot = [0, 0, 90]
    cam_3_rel_rot = [0, 0, 180]
    cam_4_rel_rot = [0, 0, 270]

    car_rot_mat = create_rot_matrix(car_rot)
    cam_2_gt_rot_mat = create_rot_matrix(cam_2_rot)
    car_cam_2_rot_mat = create_rot_matrix(car_rot) @ create_rot_matrix(cam_2_rel_rot)
    car_cam_2_rot_mat_2 = create_rot_matrix(car_rot) @ create_rot_matrix_2(cam_2_rel_rot)
    car_cam_2_rot_mat_3 = create_rot_matrix(car_rot) @ create_rot_matrix_3(cam_2_rel_rot)
    car_cam_2_rot_mat_4 = create_rot_matrix(car_rot) @ create_rot_matrix_4(cam_2_rel_rot)
    print(car_rot)


def try_videos():
    imageio.show_formats()

    ini_file = "gta-postprocessing.ini"
    visualization.multi_page = False
    visualization.ini_file = ini_file
    visualization.use_cache = False

    conn = visualization.get_connection()
    cur = conn.cursor()

    CONFIG = ConfigParser()
    CONFIG.read(ini_file)
    in_directory = r'D:\output-datasets\offroad-7'
    out_directory = r'D:\showing-videos\offroad-7'

    run_id = 4148

    cur.execute("""SELECT imagepath, \
          ARRAY[st_x(camera_relative_rotation), st_y(camera_relative_rotation), st_z(camera_relative_rotation)] as camera_relative_rotation,
          ARRAY[st_x(camera_relative_position), st_y(camera_relative_position), st_z(camera_relative_position)] as camera_relative_position 
          FROM snapshots \
          WHERE run_id = {} AND camera_fov != 0 \
          ORDER BY timestamp ASC \
        """.format(run_id))
    # camera fov is sanity check for malformed images
    results = []
    for row in cur:
        res = dict(row)
        # res['camera_relative_rotation'] = np.array(res['camera_relative_rotation'])
        results.append(res)

    print('There are {} snapshots'.format(len(results)))

    cur.execute("""SELECT DISTINCT \
          ARRAY[st_x(camera_relative_rotation), st_y(camera_relative_rotation), st_z(camera_relative_rotation)] as camera_relative_rotation, 
          ARRAY[st_x(camera_relative_position), st_y(camera_relative_position), st_z(camera_relative_position)] as camera_relative_position 
          FROM snapshots \
          WHERE run_id = {} AND camera_fov != 0 \
          ORDER BY camera_relative_rotation ASC \
        """.format(run_id))
    # camera fov is sanity check for malformed images
    print('there are following relative camera rotations')
    cam_configurations = []
    for row in cur:
        print(row['camera_relative_rotation'])
        print(row['camera_relative_position'])
        cam_configurations.append((row['camera_relative_rotation'], row['camera_relative_position']))

    def split_results_by_relative_cam_configurations(results):
        res_groups = {}
        for cam_conf in cam_configurations:
            res_groups[str(cam_conf)] = [i for i in results if
                                         (i['camera_relative_rotation'], i['camera_relative_position']) == cam_conf]
        return res_groups

    def result_group_to_video(results, suffix):
        img_sequence = [os.path.join(in_directory, i['imagepath'] + suffix) for i in results]
        clip = ImageSequenceClip(img_sequence, fps=10)
        return clip

    def process_depth_image(image):
        image = np.array(image.convert('RGB')) / np.iinfo(np.uint16).max
        return image

    def result_depth_group_to_video(results, suffix):
        img_sequence = [os.path.join(in_directory, i['imagepath'] + suffix) for i in results]
        # img_sequence = [Image.open(i) for i in img_sequence]
        # workers = 8
        # img_sequence = Parallel(n_jobs=workers, backend='threading')(delayed(process_depth_image)(i) for i in img_sequence)
        clip = ImageSequenceClip(img_sequence, fps=10, with_mask=False)
        # clip = clip.fl_image(process_depth_image)
        return clip

    def process_stencil_image(colors, image):
        image = np.array(image) % 8
        image = colors[image]
        return image

    def result_stencil_group_to_video(results, suffix):
        colors = (plt.cm.viridis(np.linspace(0, 1, 8))[:, :3] * np.iinfo(np.uint8).max).astype(np.uint8)
        img_sequence = [os.path.join(in_directory, i['imagepath'] + suffix) for i in results]
        # img_sequence = [Image.open(i) for i in img_sequence]  # IO operation, no need to perallelize
        # from 60 to 16 seconds sppedup, nice, all cores at full load
        # workers = 8
        # img_sequence = Parallel(n_jobs=workers, backend='threading')(delayed(process_stencil_image)(colors, i) for i in img_sequence)
        clip = ImageSequenceClip(img_sequence, fps=10, with_mask=False)
        # clip = clip.fl_image(process_stencil_image)
        return clip

    result_groups = split_results_by_relative_cam_configurations(results)
    for cam_conf, res in result_groups.items():
        print(cam_conf)
        print(len(res))

    for cam_conf, res in result_groups.items():
        clip = result_group_to_video(res, '.jpg')
        depth_clip = result_depth_group_to_video(res, '-depth.png')
        stencil_clip = result_stencil_group_to_video(res, '-stencil.png')

        video_name = os.path.join(out_directory, "camera-{}.mp4".format(cam_conf))
        clip.write_videofile(video_name, audio=False, codec='mpeg4', threads=8)

        video_name = os.path.join(out_directory, "camera-{}-depth.mp4".format(cam_conf))
        depth_clip.write_videofile(video_name, audio=False, codec='mpeg4')

        video_name = os.path.join(out_directory, "camera-{}-stencil.mp4".format(cam_conf))
        stencil_clip.write_videofile(video_name, audio=False, codec='mpeg4')


def try_scene_to_pointcloud():
    ini_file = "gta-postprocessing.ini"
    visualization.multi_page = False
    visualization.ini_file = ini_file
    visualization.use_cache = False

    def process_frame_tiff(directory, file_name, csv_name):
        rgb_file = os.path.join(visualization.get_in_directory(), '{}.tiff'.format(file_name))
        depth_file = os.path.join(directory, '{}-depth.png'.format(file_name))
        json_file = os.path.join(directory, '{}.json'.format(file_name))

        rgb = np.array(Image.open(rgb_file))
        # depth = load_depth(file_name)
        depth = np.array(Image.open(depth_file))
        depth = depth / np.iinfo(np.uint16).max  # normalizing into NDC

        with open(json_file, mode='r') as f:
            data = json.load(f)
        # print(data['scene_id'])

        view_matrix = np.array(data['view_matrix'])
        proj_matrix = np.array(data['proj_matrix'])

        # conn = visualization.get_connection()
        # cur = conn.cursor()
        # cur.execute("""SELECT scene_id
        #     FROM snapshots_view
        #     WHERE imagepath = %(imagepath)s
        #     """, {'imagepath': file_name})
        #
        # scene_id = cur.fetchone()['scene_id']
        #
        # cur = conn.cursor()
        # cur.execute("""SELECT camera_rot, player_pos
        #             FROM snapshots_view
        #             WHERE scene_id = %(scene_id)s AND camera_relative_rotation = ARRAY[0,0,0]::double precision[]
        #             LIMIT 1
        #             """, {'scene_id': scene_id})
        #
        # result = cur.fetchone()
        # car_rotation = np.array(result['camera_rot'])
        # car_position = np.array(result['player_pos'])
        #
        # calc_cam_position = car_and_relative_cam_to_absolute_cam_position(car_position, car_rotation, data['camera_relative_position'])
        # calc_cam_rotation = car_and_relative_cam_to_absolute_cam_rotation_angles(car_rotation, data['camera_relative_rotation'])
        #
        # view_matrix = construct_view_matrix(calc_cam_position, calc_cam_rotation)
        # proj_matrix = np.array(data['proj_matrix'])

        depth_to_csv(depth, data, csv_name, view_matrix, proj_matrix)

    def depth_to_csv(depth, data, csv_name, view_matrix, proj_matrix):
        vecs, _ = points_to_homo(data, depth, tresholding=False)
        vecs_p = ndc_to_view(vecs, proj_matrix)
        vecs_p_world = view_to_world(vecs_p, view_matrix)
        save_csv(vecs_p_world[0:3, :].T, csv_name)

    process_frame_tiff(r'D:\output-datasets\offroad-7\0', '2018-08-13--11-15-01--499', 'my-points-0-png')
    process_frame_tiff(r'D:\output-datasets\offroad-7\1', '2018-08-13--11-15-01--860', 'my-points-1-png')
    process_frame_tiff(r'D:\output-datasets\offroad-7\2', '2018-08-13--11-15-02--407', 'my-points-2-png')
    process_frame_tiff(r'D:\output-datasets\offroad-7\3', '2018-08-13--11-15-02--672', 'my-points-3-png')
    process_frame_tiff(r'D:\output-datasets\offroad-7\4', '2018-08-13--11-15-03--058', 'my-points-4-png')
    process_frame_tiff(r'D:\output-datasets\offroad-7\5', '2018-08-13--11-15-03--455', 'my-points-5-png')


if __name__ == '__main__':
    # in_directory = r'D:\projekty\GTA-V-extractors\traffic-camera-dataset\raw'
    # out_directory = r'D:\projekty\GTA-V-extractors\traffic-camera-dataset\bboxes'
    # out_2_directory = r'D:\projekty\GTA-V-extractors\traffic-camera-dataset\semantic-segmentation'
    # in_directory = r'D:\projekty\GTA-V-extractors\sample-images'
    # out_directory = r'D:\projekty\GTA-V-extractors\sample-images\output'
    #
    # pattern = '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]--[0-9][0-9]-[0-9][0-9]-[0-9][0-9]--[0-9][0-9][0-9].jpg'
    # files = glob.glob(os.path.join(in_directory, pattern))
    # print('there are {} files'.format(len(files)))
    # # name = files[0]
    # # base_name = get_base_name(name)
    # base_name = '2018-03-30--02-02-25--188'

    in_directory = r'D:\output-datasets\onroad-1'
    base_name = '2018-08-13--21-05-07--241'
    # draw3dbboxes(in_directory, base_name)
    # try_one_car_3dbboxes(in_directory, base_name)

    # proj_matrix = np.array([[1.21006660e+00, 0.00000000e+00, 0.00000000e+00,
    #                          0.00000000e+00],
    #                         [0.00000000e+00, 2.14450692e+00, 0.00000000e+00,
    #                          0.00000000e+00],
    #                         [0.00000000e+00, 0.00000000e+00, 1.49965283e-04,
    #                          1.50022495e+00],
    #                         [0.00000000e+00, 0.00000000e+00, -1.00000000e+00,
    #                          0.00000000e+00]])
    #
    # z_meters_min = 1.5
    # z_meters_max = 25
    #
    # bool_grid = np.load('../sample-images/2018-03-07--16-30-26--642.npy')
    # ndc_points = convert_bool_grid_to_ndc_pointcloud(bool_grid, proj_matrix, z_meters_min, z_meters_max)
    # ndc_points = np.hstack((ndc_points, np.ones((ndc_points.shape[0], 1)))).T
    # view_points = ndc_to_view(ndc_points, proj_matrix)
    # save_pointcloud_csv(view_points.T[:, 0:3], '{}/{}.csv'.format('../sample-images', '2018-03-07--16-30-26--642'))

    # try_points_to_grid_and_back()
    # try_subsampling()
    # try_pcl_subsampling_detailed()
    # try_simple_pointcloud_load_and_merge()
    # play_with_matrices()
    # try_videos()
    try_scene_to_pointcloud()
