import numpy as np
import os
from PIL import Image
import json

from tifffile import tifffile

from gta_math import vfov_to_hfov


def get_kitti_vfov():
    # gets kitti color camera vertical field of view, calculated from inspecting-kitti-calibration.ipynb
    return 54.9239


def get_kitti_hfov():
    # gets kitti color camera horizontal field of view, calculated from inspecting-kitti-calibration.ipynb
    return 119.6913


def get_kitti_img_size():
    return 1242, 375


def get_img_size_for_kitti_fov(data):
    vfov = data['camera_fov']
    hfov = vfov_to_hfov(data['height'], data['width'], vfov)

    gta_width = data['width']
    gta_height = data['height']

    z_width = (gta_width / 2) / np.tan(np.radians(hfov) / 2)
    kitti_width = 2 * z_width * np.tan(np.radians(get_kitti_hfov()) / 2)

    z_height = (gta_height / 2) / np.tan(np.radians(vfov) / 2)
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
    return img.crop((offset_width, offset_height, width - offset_width, height - offset_height))


def image_gta_to_kitti(rgb, depth, stencil, data):
    kitti_height, kitti_width = get_img_size_for_kitti_fov(data)
    kitti_orig_width, kitti_orig_height = get_kitti_img_size()
    # image is cropped so FOV corresponds
    # then, image is resized so actual sizes correspond
    rgb_new = crop_image_middle(rgb, kitti_height, kitti_width)
    rgb_resized = rgb_new.resize([kitti_orig_width, kitti_orig_height], Image.ANTIALIAS)
    depth_new = crop_image_middle(depth, kitti_height, kitti_width)
    depth_resized = depth_new.resize([kitti_orig_width, kitti_orig_height], Image.ANTIALIAS)
    stencil_new = crop_image_middle(stencil, kitti_height, kitti_width)
    stencil_resized = stencil_new.resize([kitti_orig_width, kitti_orig_height], Image.ANTIALIAS)
    pass


def try_image_to_kitti():
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

    image_gta_to_kitti(rgb, depth, stencil, data)


def load_kitti_calib_data(path):
    pass


if __name__ == '__main__':
    # load_kitti_calib_data('')
    try_image_to_kitti()