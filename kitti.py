import numpy as np
import os
from PIL import Image
import json
from tifffile import tifffile
from gta_math import vfov_to_hfov, construct_proj_matrix, proj_matrix_to_near_clip, points_to_homo, ndc_to_view, \
    view_to_world
import os.path as osp


def get_kitti_vfov():
    # gets kitti color camera vertical field of view, calculated from inspecting-kitti-calibration.ipynb
    # return 54.9239
    return 29.04


def get_kitti_hfov():
    # gets kitti color camera horizontal field of view, calculated from inspecting-kitti-calibration.ipynb
    # return 119.6913
    return 81.37


def get_kitti_img_size():
    return 1242, 375


def get_img_size_for_kitti_fov(width, height, fov):
    vfov = fov
    hfov = vfov_to_hfov(height, width, vfov)

    z_width = (width / 2) / np.tan(np.radians(hfov) / 2)
    kitti_width = 2 * z_width * np.tan(np.radians(get_kitti_hfov()) / 2)

    z_height = (height / 2) / np.tan(np.radians(vfov) / 2)
    kitti_height = 2 * z_height * np.tan(np.radians(get_kitti_vfov()) / 2)

    kitti_ratio = kitti_width / kitti_height    # this fits with the kitti width to height ratio in pixels

    kitti_orig_width, kitti_orig_height = get_kitti_img_size()
    kitti_orig_ratio = kitti_orig_width / kitti_orig_height

    # just sanity check
    assert np.isclose(kitti_orig_ratio, kitti_ratio, atol=1e-2)
    return kitti_height, kitti_width


def crop_image_middle(img, height, width):
    orig_height, orig_width = img.shape[:2]
    img = Image.fromarray(img)
    offset_height = (orig_height - height) / 2
    offset_width = (orig_width - width) / 2
    return img.crop((offset_width, offset_height, orig_width - offset_width, orig_height - offset_height))


def image_gta_to_kitti(rgb, depth, stencil, width, height, fov):
    kitti_height, kitti_width = get_img_size_for_kitti_fov(width, height, fov)
    kitti_orig_width, kitti_orig_height = get_kitti_img_size()
    # image is cropped so FOV corresponds
    # then, image is resized so actual sizes correspond
    rgb_new = crop_image_middle(rgb, kitti_height, kitti_width)
    depth_new = crop_image_middle(depth, kitti_height, kitti_width)
    stencil_new = crop_image_middle(stencil, kitti_height, kitti_width)
    rgb_new = rgb_new.resize([kitti_orig_width, kitti_orig_height], Image.ANTIALIAS)
    depth_new = depth_new.resize([kitti_orig_width, kitti_orig_height], Image.NEAREST)
    stencil_new = stencil_new.resize([kitti_orig_width, kitti_orig_height], Image.NEAREST)
    return np.array(rgb_new), np.array(depth_new), np.array(stencil_new)


def get_kitti_proj_matrix(proj_matrix):
    width, height = get_kitti_img_size()
    return construct_proj_matrix(height, width, get_kitti_vfov(), proj_matrix_to_near_clip(proj_matrix))


def readVariable(data=None, name=None, M=None, N=None):
    if name not in data:
        return []

    if M != 1 or N != 1:
        values = np.array(data[name].split(), dtype=float)
        values = values.reshape(M, N)
        return values
    else:
        return data[name]


def isempty(a):
    try:
        return 0 in np.asarray(a).shape
    except AttributeError:
        return False


def load_calibration_cam_to_cam(filename):
    # open file
    with open(filename, 'r') as stream:
        import ruamel.yaml as yaml
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return {}

    calib = {'cornerdist': readVariable(data, 'corner_dist', 1, 1),
             'S': np.zeros((4, 1, 2)),
             'K': np.zeros((4, 3, 3)),
             'D': np.zeros((4, 1, 5)),
             'R': np.zeros((4, 3, 3)),
             'T': np.zeros((4, 3, 1)),
             'S_rect': np.zeros((4, 1, 2)),
             'R_rect': np.zeros((4, 3, 3)),
             'P_rect': np.zeros((4, 3, 4))
             }

    # read corner distance
    # /opt/project/devkit/matlab/load_calibration_cam_to_cam.m:12
    # read all cameras (maximum: 100)

    for cam in np.array(range(4)).reshape(-1):
        # read variables
        S_ = readVariable(data, 'S_{:02d}'.format(cam), 1, 2)
        K_ = readVariable(data, 'K_{:02d}'.format(cam), 3, 3)
        D_ = readVariable(data, 'D_{:02d}'.format(cam), 1, 5)
        R_ = readVariable(data, 'R_{:02d}'.format(cam), 3, 3)
        T_ = readVariable(data, 'T_{:02d}'.format(cam), 3, 1)
        S_rect_ = readVariable(data, 'S_rect_{:02d}'.format(cam), 1, 2)
        R_rect_ = readVariable(data, 'R_rect_{:02d}'.format(cam), 3, 3)
        P_rect_ = readVariable(data, 'P_rect_{:02d}'.format(cam), 3, 4)
        if isempty(S_) or isempty(K_) or isempty(D_) or isempty(R_) or isempty(T_):
            break
        # write calibration
        calib['S'][cam] = S_
        calib['K'][cam] = K_
        calib['D'][cam] = D_
        calib['R'][cam] = R_
        calib['T'][cam] = T_

        if (not isempty(S_rect_)) and (not isempty(R_rect_)) and (not isempty(P_rect_)):
            calib['S_rect'][cam] = S_rect_
            calib['R_rect'][cam] = R_rect_
            calib['P_rect'][cam] = P_rect_

    return calib


def check_proj_matrices(depth, data):
    calib = load_calibration_cam_to_cam('kitti-calibration-example/calib_cam_to_cam.txt')
    kitti_proj_matrix = calib['P_rect'][2]
    proj_matrix = np.array(data['proj_matrix'])
    cropped_proj_matrix = proj_matrix[(0, 1, 3), :]
    width, height = get_kitti_img_size()
    to_pixel_matrix = np.array([
        [width/2, 0, width/2],
        [0, -height/2, height/2],
        [0, 0, 1],
    ])
    result = to_pixel_matrix @ cropped_proj_matrix

    data['cam_far_clip'] = 1000
    points_ndc, pixels = points_to_homo(data, depth, tresholding=True)
    points_meters = ndc_to_view(points_ndc, proj_matrix)
    points_meters[2, :] *= -1

    points_pixels = np.zeros((3, points_ndc.shape[1]))
    points_pixels[0] = pixels[1] * points_meters[2, :]
    points_pixels[1] = pixels[0] * points_meters[2, :]
    points_pixels[2] = points_meters[2, :]
    # points_pixels must contain point values after the normalization (dividing by depth),
    # so now they must be multiplied by depth

    points_meters = points_meters[0:3, :]

    calculated_kitti_proj, residuals, rank, s = np.linalg.lstsq(points_meters.T, points_pixels.T)
    calculated_kitti_proj = calculated_kitti_proj.T  # need to transpose it because of matrix multiplication from the other side
    pass


def try_image_to_kitti():
    # loading data
    directory = r'D:\output-datasets\offroad-14\1'
    base_name = '000084'
    rgb_file = os.path.join(directory, '{}.jpg'.format(base_name))
    depth_file = os.path.join(directory, '{}-depth.tiff'.format(base_name))
    stencil_file = os.path.join(directory, '{}-stencil.png'.format(base_name))
    json_file = os.path.join(directory, '{}.json'.format(base_name))
    rgb = np.array(Image.open(rgb_file))
    depth = tifffile.imread(depth_file)
    stencil = np.array(Image.open(stencil_file))
    with open(json_file) as f:
        data = json.load(f)

    # creating pointcloud for original data
    csv_name = base_name + '-orig'
    vecs, _ = points_to_homo(data, depth, tresholding=False)
    vecs_p = ndc_to_view(vecs, data['proj_matrix'])
    vecs_p_world = view_to_world(vecs_p, np.array(data['view_matrix']))
    a = np.asarray(vecs_p_world[0:3, :].T)
    np.savetxt(os.path.join('kitti-format', "points-{}.csv".format(csv_name)), a, delimiter=",")

    # whole gta to kitti transformation
    rgb, depth, stencil = image_gta_to_kitti(rgb, depth, stencil, data['width'], data['height'], data['camera_fov'])
    data['proj_matrix'] = get_kitti_proj_matrix(np.array(data['proj_matrix'])).tolist()
    data['width'], data['height'] = get_kitti_img_size()

    # saving new images
    Image.fromarray(rgb).convert(mode="RGB").save(os.path.join('kitti-format', '{}.jpg'.format(base_name)))
    tifffile.imsave(os.path.join('kitti-format', '{}-depth-orig.tiff'.format(base_name)), depth)
    tifffile.imsave(os.path.join('kitti-format', '{}-depth-lzma.tiff'.format(base_name)), depth, compress='lzma')
    tifffile.imsave(os.path.join('kitti-format', '{}-depth-zip-5.tiff'.format(base_name)), depth, compress=5)
    tifffile.imsave(os.path.join('kitti-format', '{}-depth-zip-9.tiff'.format(base_name)), depth, compress=9)
    tifffile.imsave(os.path.join('kitti-format', '{}-depth-zip-zstd.tiff'.format(base_name)), depth, compress='zstd')
    Image.fromarray(stencil).save(os.path.join('kitti-format', '{}-stencil.jpg'.format(base_name)))
    with open(os.path.join('kitti-format', '{}.json'.format(base_name)), 'w+') as f:
        json.dump(data, f)

    data['view_matrix'] = np.array(data['view_matrix'])

    check_proj_matrices(depth, data)

    # creating pointcloud for kitti format data
    csv_name = base_name + '-kitti'
    vecs, _ = points_to_homo(data, depth, tresholding=False)
    vecs_p = ndc_to_view(vecs, data['proj_matrix'])
    vecs_p_world = view_to_world(vecs_p, np.array(data['view_matrix']))
    a = np.asarray(vecs_p_world[0:3, :].T)
    np.savetxt(os.path.join('kitti-format', "points-{}.csv".format(csv_name)), a, delimiter=",")


if __name__ == '__main__':
    # load_kitti_calib_data('')
    try_image_to_kitti()
