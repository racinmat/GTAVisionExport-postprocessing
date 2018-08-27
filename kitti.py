import numpy as np


def get_kitti_vfov():
    # gets kitti color camera vertical field of view, calculated from inspecting-kitti-calibration.ipynb
    return 54.63


def get_kitti_hfov():
    # gets kitti color camera horizontal field of view, calculated from inspecting-kitti-calibration.ipynb
    return 119.64


def image_gta_to_kitti(data):
    d = (data['width']/2) / np.sin(np.radians(data['fov']/2))
    kitti_width = d * np.sin(np.radians(get_kitti_hfov()))

    d = (data['height']/2) / np.sin(np.radians(data['fov']/2))
    kitti_height = d * np.sin(np.radians(get_kitti_vfov()))
    pass  # todo: implement


def load_kitti_calib_data(path):
    pass


if __name__ == '__main__':
    load_kitti_calib_data('')