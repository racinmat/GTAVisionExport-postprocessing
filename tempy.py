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
from gta_math import *
from visualization import save_pointcloud_csv
from voxelmaps import convert_ndc_pointcloud_to_bool_grid


def get_base_name(name):
    return os.path.basename(os.path.splitext(name)[0])


def test_plane(X0, X1, X2, x, y, z):
    v1 = X1 - X0
    v2 = X2 - X0
    n = np.cross(v1, v2)

    return n[0] * (x - X0[0]) + n[1] * (y - X0[1]) + n[2] * (z - X0[2]) > 0


def draw3dbboxes(in_directory, out_directory, base_name):
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
                    e['type'] == 'car' and e['class'] != 'Trains' and is_entity_in_image(depth, e, view_matrix,
                                                                                         proj_matrix, width, height)]

    print('camera pos: ', data['camera_pos'])
    fig = plt.figure(figsize=(16, 10))
    plt.axis('off')
    #    plt.xlim([0, rgb.shape[1]])
    #    plt.ylim([rgb.shape[0], 0])
    ax = plt.gca()
    plt.imshow(rgb)

    for row in visible_cars:
        row['bbox_calc'] = calculate_2d_bbox(row, view_matrix, proj_matrix, width, height)
        is_entity_in_image(depth, row, view_matrix, proj_matrix, width, height)
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
                    e['class'] != 'Trains' and is_entity_in_image(depth, e, view_matrix, proj_matrix, width, height)]

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
        row['bbox_calc'] = calculate_2d_bbox(row, view_matrix, proj_matrix, width, height)
        # position in world coords
        pos = np.array(row['pos'])
        rot = np.array(row['rot'])

        model_sizes = np.array(row['model_sizes'])
        points_3dbbox = get_model_3dbbox(model_sizes)

        # 3D bounding box
        # projecting cuboid to 2d
        bbox_3d = model_coords_to_world(pos, rot, points_3dbbox, view_matrix, proj_matrix, width, height)
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
        in1 = test_plane(bbox_3d[3, :], bbox_3d[2, :], bbox_3d[7, :], x, y, z)
        in2 = test_plane(bbox_3d[1, :], bbox_3d[5, :], bbox_3d[0, :], x, y, z)
        in3 = test_plane(bbox_3d[6, :], bbox_3d[2, :], bbox_3d[4, :], x, y, z)
        in4 = test_plane(bbox_3d[3, :], bbox_3d[7, :], bbox_3d[1, :], x, y, z)
        in5 = test_plane(bbox_3d[7, :], bbox_3d[6, :], bbox_3d[5, :], x, y, z)
        in6 = test_plane(bbox_3d[0, :], bbox_3d[2, :], bbox_3d[1, :], x, y, z)
        is_inside = in1 & in2 & in3 & in4 & in5 & in6

        rgb[(idxs[0][is_inside], idxs[1][is_inside])] = np.array(cm.viridis(id * 25)[0:3]) * 255

        id += 1
        # assuming there are no more than 10 cars in image
    #    plt.imshow(car_id_mask)
    #    plt.clim(0.1, 1)
    plt.imshow(rgb)
    # plt.savefig('{}/{}.jpg'.format(out_directory, base_name), bbox_inches='tight')
    plt.show()


def test_points_to_grid_and_back():
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
    ndc_grid = convert_ndc_pointcloud_to_bool_grid(x_range, y_range, z_range, ndc_points, proj_matrix, z_meters_min, z_meters_max)
    ndc_points_reconst = convert_bool_grid_to_ndc_pointcloud(ndc_grid, proj_matrix, z_meters_min, z_meters_max)
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
    # draw3dbboxes(in_directory, out_directory, base_name)

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

    test_points_to_grid_and_back()