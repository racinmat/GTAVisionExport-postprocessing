import os
from configparser import ConfigParser
import numpy as np
import re
from PIL import Image, ImageFile
from skimage import io
from matplotlib import cm, patches
import matplotlib.pyplot as plt
import psycopg2
import tifffile
from psycopg2.extras import DictCursor
from psycopg2.extensions import connection
# threaded connection pooling
from psycopg2.pool import PersistentConnectionPool
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def get_connection():
    """
    :rtype: connection
    """
    global conn
    if conn is None:
        CONFIG = ConfigParser()
        CONFIG.read(ini_file)
        conn = psycopg2.connect(CONFIG["Postgres"]["db"], cursor_factory=DictCursor)
    return conn


def get_connection_pooled():
    """
    :rtype: connection
    """
    global conn_pool
    if conn_pool is None:
        CONFIG = ConfigParser()
        CONFIG.read(ini_file)
        conn_pool = PersistentConnectionPool(conn_pool_min, conn_pool_max, CONFIG["Postgres"]["db"], cursor_factory=DictCursor)
    conn = conn_pool.getconn()
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn


def get_gta_image_jpg_dir():
    return '/datagrid/personal/racinmat/GTA-jpg'


def bbox_from_string(string):
    return np.array([float(i) for i in re.sub('[()]', '', string).split(',')]).reshape(2, 2)


def get_bounding_boxes(name):
    return get_detections(name, "AND NOT bbox @> POINT '(Infinity, Infinity)'")


def get_detections(name, additional_condition = ''):
    name = name.replace('info-', '')
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""SELECT bbox, ARRAY[st_x(pos), st_y(pos), st_z(pos)] as pos,
        ARRAY[st_xmin(bbox3d), st_xmax(bbox3d), st_ymin(bbox3d), st_ymax(bbox3d), st_zmin(bbox3d), st_zmax(bbox3d)] as bbox3d, 
         type, class, handle, detections.snapshot_id
        FROM detections
        JOIN snapshots ON detections.snapshot_id = snapshots.snapshot_id
        WHERE imagepath = '{}' {}
        """.format(name, additional_condition))
    # print(size)
    results = []
    for row in cur:
        res = dict(row)
        res['bbox3d'] = np.array(res['bbox3d'])
        res['bbox'] = bbox_from_string(res['bbox'])
        res['pos'] = np.array(res['pos'])
        results.append(res)
    return results


def get_detections_multiple(names, additional_condition=''):
    # this is batch operation for multiple files
    names_string = ', '.join(names)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""SELECT bbox, ARRAY[st_x(pos), st_y(pos), st_z(pos)] as pos,
        ARRAY[st_xmin(bbox3d), st_xmax(bbox3d), st_ymin(bbox3d), st_ymax(bbox3d), st_zmin(bbox3d), st_zmax(bbox3d)] as bbox3d, 
        type, class, handle, detections.snapshot_id, imagepath
        FROM detections
        JOIN snapshots ON detections.snapshot_id = snapshots.snapshot_id
        WHERE imagepath = ANY(ARRAY[{}]) {}
        ORDER BY detection_id ASC
        """.format(names_string, additional_condition))
    # print(size)
    results = []
    for row in cur:
        res = dict(row)
        res['bbox3d'] = np.array(res['bbox3d'])
        res['bbox'] = bbox_from_string(res['bbox'])
        res['pos'] = np.array(res['pos'])
        results.append(res)
    return results


def show_loaded_bounding_boxes(detections, size, ax):
    # print(size)
    for row in detections:
        # bbox format is
        # [max x, max y]
        # [min x, min y]
        bbox = np.copy(row['bbox'])
        # print(bbox)
        # bbox_x = bbox[:,0]
        # bbox_y = bbox[:,1]
        bbox[:, 0] *= size[1]
        bbox[:, 1] *= size[0]
        # print(bbox)

        width, height = bbox[0, :] - bbox[1, :]
        rect = patches.Rectangle(bbox[1, :], width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    # print(rows)
    # bbox = np.array([1, 1, 0, 0]).reshape(2, 2)
    # bbox_y = bbox[:,0]
    # bbox_x = bbox[:,1]
    # bbox[:, 0] *= size[0]
    # bbox[:, 1] *= size[1]

    # height, width = bbox[0, :] - bbox[1, :]
    # rect = patches.Rectangle(bbox[1, :], width, height, linewidth=3, edgecolor='y', facecolor='none')

    # Add the patch to the Axes
    # ax.add_patch(rect)


def show_bounding_boxes(name, size, ax):
    detections = get_bounding_boxes(name)
    show_loaded_bounding_boxes(detections, size, ax)


def load_depth(name):
    if name not in depths:
        if multi_page:
            tiff_depth = Image.open(os.path.join(get_in_directory(), name + '.tiff'))
            tiff_depth.seek(2)
        else:
            tiff_depth = tifffile.imread(os.path.join(get_in_directory(), name + '-depth.tiff'))
        if use_cache:
            depths[name] = tiff_depth
        else:
            return tiff_depth
    return depths[name]


def load_stencil(name):
    if name not in stencils:
        if multi_page:
            tiff_stencil = Image.open(os.path.join(get_in_directory(), name + '.tiff'))
            tiff_stencil.seek(1)
            tiff_stencil = np.array(tiff_stencil)
        else:
            tiff_stencil = tifffile.imread(os.path.join(get_in_directory(), name + '-stencil.tiff'))
        if use_cache:
            stencils[name] = tiff_stencil
        else:
            return tiff_stencil
    return stencils[name]


def load_stencil_ids(name):
    stencil = load_stencil(name)
    return stencil % 16  # only last 4 bits are object ids


def load_stencil_flags(name):
    stencil = load_stencil(name)
    return stencil - (stencil % 16)  # only first 4 bits are flags


def ids_to_greyscale(arr):
    # there are 4 bits -> 16 values for arrays, transfer from range [0-15] to range [0-255]
    return arr * 4


def show_bboxes(name):
    im = Image.open(os.path.join(get_in_directory(), name + '.tiff'))
    size = (im.size[1], im.size[0])
    fig = plt.figure()
    plt.imshow(im)
    show_bounding_boxes(name, size, plt.gca())
    plt.savefig(os.path.join(out_directory, 'bboxes-' + name + '.jpg'))


def main():
    files = [
        # 'info-2017-11-14--17-48-28',
        # 'info-2017-11-14--17-48-32',
        # 'info-2017-11-14--17-48-34',
        # 'info-2017-11-14--12-06-33',
        # 'info-2017-11-14--12-03-12',
        # 'info-2017-11-14--12-03-15',
        # 'info-2017-11-14--12-03-17',
        # 'info-2017-11-14--17-49-23',
        # 'info-2017-11-14--17-48-41',
        # 'info-2017-11-14--17-48-43',
        # 'info-2017-11-14--17-48-44',
        # 'info-2017-11-14--17-48-47',
        # 'info-2017-11-19--23-21-08',
        # 'info-2017-11-19--23-20-45',
        # 'info-2017-11-19--23-20-49',
        # 'info-2017-11-24--18-48-25--561'
        'single-res'
        # 'single'
    ]
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for name in files:
        im = Image.open(os.path.join(get_in_directory(), name + '.tiff'))
        size = (im.size[1], im.size[0])

        # show_bboxes(name)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        plt.tight_layout()

        plt.axis('off')
        im.seek(0)
        ax1.imshow(im)
        # ax2.set_title('f')
        ax2.imshow(ids_to_greyscale(load_stencil_ids(name)), cmap='gray')
        ax3.set_title('ids')
        # ax3.imshow(load_stencil_ids(name), cmap='gray')
        ax3.imshow(ids_to_greyscale(load_stencil_ids(name)), cmap='plasma')
        ax4.set_title('depth')
        ax4.imshow(load_depth(name), cmap='gray')
        show_bounding_boxes(name, size, ax1)
        show_bounding_boxes(name, size, ax3)
        plt.axis('off')
        # plt.draw()
        plt.show()

    plt.show()


multi_page = True
# multi_page = False

depths = {}
stencils = {}
ini_file = "gta-postprocessing.ini"
in_directory = None
out_directory = './img'
conn = None
conn_pool = None
conn_pool_min = 1
conn_pool_max = 28
use_cache = True    # for depth and stencil cache. Is not usable for batch operations with big data and only eats RAM


def get_in_directory():
    global in_directory
    if in_directory is None:
        CONFIG = ConfigParser()
        CONFIG.read(ini_file)
        in_directory = CONFIG["Images"]["Tiff"]
    return in_directory


def save_pointcloud_csv(vecs, name):
    assert(vecs.shape[1] == 3)
    a = np.asarray(vecs)
    np.savetxt(name, a, delimiter=",")


if __name__ == '__main__':
    main()

# online combining to multipage tiff https://www.coolutils.com/Online/TIFF-Combine/
