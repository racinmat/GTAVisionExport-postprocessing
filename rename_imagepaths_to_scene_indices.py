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


def get_files(run_directory, camera_dir, suffix):
    return glob.glob(osp.join(run_directory, camera_dir, visualization.get_dataset_filename_wildcard() + suffix))


def main():
    parser = argparse.ArgumentParser(description='Convert filenames.')
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

    sample_file_path = osp.join(run_directory, '0')
    if not osp.exists(sample_file_path):
        print('path to dataset images can not be found, tried {}'.format(sample_file_path))
        return

    # at first, I find the run_id by checking first image name
    cur: cursor = conn.cursor()
    imagepath = get_base_name(glob.glob(osp.join(sample_file_path, '*.jpg'))[0])

    cur.execute("""SELECT run_id
          FROM snapshots \
          WHERE imagepath = %(imagepath)s
        """, {'imagepath': imagepath})

    run_id = cur.fetchone()['run_id']

    # then I get sorted all scene ids
    cur = conn.cursor()

    cur.execute("""SELECT scene_id, min(timestamp) 
            FROM snapshots
            WHERE run_id = %(run_id)s
            GROUP BY scene_id
          ORDER BY min(timestamp) ASC
        """, {'run_id': run_id})

    scenes = {}
    for i, row in enumerate(cur):
        scenes[row['scene_id']] = str(i)

    # then I take all imagepaths for all scene ids
    cur = conn.cursor()

    cur.execute("""SELECT scene_id, imagepath
          FROM snapshots \
          WHERE run_id = %(run_id)s
          ORDER BY timestamp ASC
        """, {'run_id': run_id})

    file_names = {}
    for i, row in enumerate(cur):
        file_names[row['imagepath']] = row['scene_id']

    print('everything loaded, starting renaming')
    widgets = [progressbar.Percentage(), ' ', progressbar.Counter(), ' ', progressbar.Bar(), ' ',
               progressbar.FileTransferSpeed()]

    pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(file_names) * 4).start()
    counter = 0

    # I know directory structure is root/cam_index/files
    for camera_dir in os.listdir(run_directory):
        # each of 4 file types in different for loop, for simplicity
        for suffix in ['.jpg', '-depth.png', '-stencil.png', '.json']:
            for filename in get_files(run_directory, camera_dir, suffix):
                basename = get_base_name(filename)
                base_suffix = suffix.split('.')[0]
                if len(base_suffix) > 1:
                    basename = basename[:-len(base_suffix)]
                counter += 1
                pbar.update(counter)
                old_name = osp.join(run_directory, camera_dir, basename+suffix)
                new_name = osp.join(run_directory, camera_dir, scenes[file_names[basename]]+suffix)
                os.rename(old_name, new_name)

    pbar.finish()
    print('done')


if __name__ == '__main__':
    main()
