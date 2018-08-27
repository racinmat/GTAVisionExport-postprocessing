import numpy as np
import os
from PIL import Image
import json

from tifffile import tifffile


def get_kitti_vfov():
    # gets kitti color camera vertical field of view, calculated from inspecting-kitti-calibration.ipynb
    return 54.63


def get_kitti_hfov():
    # gets kitti color camera horizontal field of view, calculated from inspecting-kitti-calibration.ipynb
    return 119.64


def image_gta_to_kitti(data):
    vfov = data['camera_fov']
    hfov = 0    # todo: implement
    d = (data['width']/2) / np.sin(np.radians(data['camera_fov']/2))
    kitti_width = d * np.sin(np.radians(get_kitti_hfov()))

    d = (data['height']/2) / np.sin(np.radians(data['camera_fov']/2))
    kitti_height = d * np.sin(np.radians(get_kitti_vfov()))
    pass  # todo: implement


def try_image_to_kitti():
    directory = r'D:\output-datasets\offroad-17\1'
    base_name = '000084'
    rgb_file = os.path.join(directory, '{}.jpg'.format(base_name))
    depth_file = os.path.join(directory, '{}-depth.tiff'.format(base_name))
    stencil_file = os.path.join(directory, '{}-stencil.png'.format(base_name))
    json_file = os.path.join(directory, '{}.json'.format(base_name))
    rgb = Image.open(rgb_file)
    depth = tifffile.imread(depth_file)
    stencil = np.array(Image.open(stencil_file))
    with open(json_file) as f:
        data = json.load(f)

    image_gta_to_kitti(data)


def load_kitti_calib_data(path):
    pass


if __name__ == '__main__':
    # load_kitti_calib_data('')
    try_image_to_kitti()