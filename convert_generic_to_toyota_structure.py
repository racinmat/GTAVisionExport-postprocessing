"""
This script renames files in dataset.
Each camera needs to have its own directory.
Then scenes are sorted by generation time,
and each image is named by scene index, 0-based.
"""
import glob

import numpy as np
import matplotlib.pyplot as plt
from psycopg2.psycopg1 import cursor

import visualization
import os
from gta_math import points_to_homo, ndc_to_view, construct_proj_matrix, view_to_world, construct_view_matrix
from visualization import load_depth, load_stencil, save_pointcloud_csv, bbox_from_string, are_buffers_same_as_previous, \
    is_first_record_in_run, camera_to_string
import progressbar
from joblib import Parallel, delayed
from configparser import ConfigParser
from PIL import Image
import pickle
import json
import time
from shutil import copyfile
from functools import lru_cache
import argparse
import os.path as osp


def get_base_name(name):
    return os.path.basename(os.path.splitext(name)[0])


def get_files(run_directory, suffix):
    return glob.glob(osp.join(run_directory, visualization.get_dataset_filename_wildcard() + suffix))


def main():
    parser = argparse.ArgumentParser(description='Convert filenames and directory structure.')
    parser.add_argument('--run_directory', '-r', metavar='MY_DATA_DIR', type=str,
                        help='root of dataset files')

    args = parser.parse_args()
    run_directory = args.run_directory

    ini_file = "gta-postprocessing.ini"
    visualization.multi_page = False
    visualization.ini_file = ini_file
    visualization.use_cache = False

    conn = visualization.get_connection_pooled()

    CONFIG = ConfigParser()
    CONFIG.read(ini_file)

    if not osp.exists(run_directory):
        print('path to dataset images can not be found, tried {}'.format(run_directory))
        return

    # at first, I find the run_id by checking first image name
    cur: cursor = conn.cursor()
    imagepath = get_base_name(glob.glob(osp.join(run_directory, '*.jpg'))[0])

    cur.execute("""SELECT run_id
          FROM snapshots \
          WHERE imagepath = %(imagepath)s
        """, {'imagepath': imagepath})

    run_id = cur.fetchone()['run_id']

    camera_names, cam_configurations = visualization.get_cameras_for_run(run_id)
    camera_aliases = {
        'camera_-0.80_0.80_0.40__0.00_0.00_90.00': 'cam_left',
        'camera_0.00_-2.30_0.30__0.00_0.00_180.00': 'cam_rear',
        'camera_0.00_2.00_0.30__0.00_0.00_0.00': 'cam_front',
        'camera_0.80_0.80_0.40__0.00_0.00_270.00': 'cam_right',
    }

    for cam_string, cam_name in camera_names.items():
        camera_names[cam_string] = camera_aliases[cam_string]

    cur = conn.cursor()
    cur.execute("""SELECT 
          ARRAY[st_x(camera_relative_rotation), st_y(camera_relative_rotation), st_z(camera_relative_rotation)] as camera_relative_rotation,
          ARRAY[st_x(camera_relative_position), st_y(camera_relative_position), st_z(camera_relative_position)] as camera_relative_position,
          imagepath
          FROM snapshots \
          WHERE run_id = %(run_id)s
          ORDER BY timestamp ASC
        """, {'run_id': run_id})

    file_to_camera = {}
    for row in cur:
        res = dict(row)
        file_to_camera[res['imagepath']] = camera_names[camera_to_string(res)]

    print('everything loaded, starting moving')
    widgets = [progressbar.Percentage(), ' ', progressbar.Counter(), ' ', progressbar.Bar(), ' ',
               progressbar.FileTransferSpeed()]

    # 3 files per image
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(file_to_camera) * 3).start() # some images are excluded, but this is upper bound
    counter = 0

    suffix_dirs = {
        '.txt': 'bboxes',
        '.jpg': 'images',
    }

    # txt and jpg are moved, .cam is same per camera, thus they are deleted and only one is kept
    for suffix in ['.txt', '.jpg']:
        for filename in get_files(run_directory, suffix):
            basename = get_base_name(filename)
            base_suffix = suffix.split('.')[0]
            if len(base_suffix) > 1:
                basename = basename[:-len(base_suffix)]
            counter += 1
            pbar.update(counter)
            old_name = osp.join(run_directory, basename + suffix)
            new_name = osp.join(run_directory, file_to_camera[basename], suffix_dirs[suffix], basename + suffix)

            if not os.path.exists(os.path.dirname(new_name)):
                os.makedirs(os.path.dirname(new_name))

            os.rename(old_name, new_name)   # moving is done also by rename

    suffix = '.cam'
    for filename in get_files(run_directory, suffix):
        basename = get_base_name(filename)
        base_suffix = suffix.split('.')[0]
        if len(base_suffix) > 1:
            basename = basename[:-len(base_suffix)]
        counter += 1
        pbar.update(counter)
        old_name = osp.join(run_directory, basename + suffix)
        new_name = osp.join(run_directory, file_to_camera[basename], file_to_camera[basename] + suffix)

        # if there is calibration matrix, just delete the old one
        if os.path.exists(new_name):
            os.remove(old_name)
        else:
            os.rename(old_name, new_name)   # moving is done also by rename

    pbar.finish()
    print('done')


if __name__ == '__main__':
    main()
