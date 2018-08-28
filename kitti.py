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
    return 54.9239


def get_kitti_hfov():
    # gets kitti color camera horizontal field of view, calculated from inspecting-kitti-calibration.ipynb
    return 119.6913


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

    # just sanity check
    assert np.isclose(kitti_orig_width / kitti_orig_height, kitti_ratio)
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


def try_image_to_kitti():
    # loading data
    directory = r'D:\output-datasets\offroad-17\1'
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
    tifffile.imsave(os.path.join('kitti-format', '{}-depth.tiff'.format(base_name)), depth, compress='lzma')
    Image.fromarray(stencil).save(os.path.join('kitti-format', '{}-stencil.jpg'.format(base_name)))
    with open(os.path.join('kitti-format', '{}.json'.format(base_name)), 'w+') as f:
        json.dump(data, f)

    data['view_matrix'] = np.array(data['view_matrix'])

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

