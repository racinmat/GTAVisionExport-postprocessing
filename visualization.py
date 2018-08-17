import os
from configparser import ConfigParser
from functools import lru_cache
import numpy as np
import re
from PIL import Image, ImageFile, ImageDraw
from skimage import io
from matplotlib import cm, patches, colors
import matplotlib.pyplot as plt
import psycopg2
import tifffile
from psycopg2.extras import DictCursor
from psycopg2.extensions import connection
# threaded connection pooling
from psycopg2.pool import PersistentConnectionPool
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from gta_math import construct_view_matrix, construct_proj_matrix, points_to_homo, ndc_to_view, view_to_world, \
    is_entity_in_image, calculate_2d_bbox, world_coords_to_pixel, get_model_3dbbox, model_coords_to_pixel, \
    construct_model_matrix
from matplotlib.pyplot import Figure
from numpy import ndarray


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


def get_car_positions(handle, run_id, snapshot_id=None, offset=None):
    conn = get_connection()
    cur = conn.cursor()
    if snapshot_id is None and offset is None:
        cur.execute("""SELECT ARRAY[st_x(pos), st_y(pos), st_z(pos)] as pos
            FROM detections
            JOIN snapshots s2 on detections.snapshot_id = s2.snapshot_id
            WHERE handle = {} and run_id = {}
            ORDER BY created
            """.format(handle, run_id))
    else:
        # only positions [offset] before and after some snapshot id
        assert type(offset) == int
        assert type(snapshot_id) == int
        cur.execute("""SELECT ARRAY[st_x(pos), st_y(pos), st_z(pos)] as pos
            FROM detections
            JOIN snapshots s2 on detections.snapshot_id = s2.snapshot_id
            WHERE handle = {handle} and run_id = {run_id} 
            and s2.snapshot_id < ({snapshot_id} + {offset}) and s2.snapshot_id > ({snapshot_id} - {offset})  
            ORDER BY created
            """.format(handle=handle, run_id=run_id, snapshot_id=snapshot_id, offset=offset))

    # print(size)
    results = []
    for row in cur:
        res = dict(row)
        results.append(res['pos'])
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


def draw3dbboxes(rgb, depth, stencil, data, fig):
    """
    :param rgb:
    :type rgb: ndarray
    :param depth:
    :type depth: ndarray
    :param data:
    :type data: dict
    :param fig:
    :type fig: Figure
    """
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
    ax = fig.gca()
    ax.axis('off')
    ax.imshow(rgb)

    for row in visible_cars:
        draw_one_entity_3dbbox(row, view_matrix, proj_matrix, width, height, ax)


def draw3dbboxes_pillow(rgb, depth, stencil, data):
    """
    :param rgb:
    :type rgb: ndarray
    :param depth:
    :type depth: ndarray
    :param stencil:
    :type stencil: ndarray
    :param data:
    :type data: dict
    :param fig:
    :type fig: Figure
    """
    entities = data['entities']
    view_matrix = np.array(data['view_matrix'])
    proj_matrix = np.array(data['proj_matrix'])
    width = data['width']
    height = data['height']
    # visible_cars = [e for e in entities if e['bbox'][0] != [np.inf, np.inf] and e['type'] == 'car']
    visible_cars = [e for e in entities if
                    e['type'] == 'car' and e['class'] != 'Trains' and is_entity_in_image(depth, stencil, e, view_matrix,
                                                                                         proj_matrix, width, height)]

    im = Image.fromarray(rgb)
    for row in visible_cars:
        draw_one_entity_3dbbox_pillow(row, view_matrix, proj_matrix, width, height, im)
    return im


def calculate_one_entity_bbox(row, view_matrix, proj_matrix, width, height):
    row['bbox_calc'] = calculate_2d_bbox(row, view_matrix, proj_matrix, width, height)
    pos = np.array(row['pos'])

    bbox = np.array(row['bbox_calc'])
    bbox[:, 0] *= width
    bbox[:, 1] *= height

    # 3D bounding box
    rot = np.array(row['rot'])
    model_sizes = np.array(row['model_sizes'])
    points_3dbbox = get_model_3dbbox(model_sizes)

    # projecting cuboid to 2d
    bbox_2d = model_coords_to_pixel(pos, rot, points_3dbbox, view_matrix, proj_matrix, width, height).T
    # print('3D bbox:\n', points_3dbbox)
    # print('3D bbox in 2D:\n', bbox_2d)
    return bbox, bbox_2d


def draw_one_entity_3dbbox(row, view_matrix, proj_matrix, width, height, ax):
    bbox, bbox_2d = calculate_one_entity_bbox(row, view_matrix, proj_matrix, width, height)
    # showing cuboid
    draw_one_entity_3dbbox_matplotlib(bbox, bbox_2d, ax)


def draw_one_entity_3dbbox_matplotlib(bbox, bbox_2d, ax):
    bbox_width, bbox_height = bbox[0, :] - bbox[1, :]
    print('2D bbox:', bbox)
    rect = patches.Rectangle(bbox[1, :], bbox_width, bbox_height, linewidth=1, edgecolor='y', facecolor='none')
    ax.add_patch(rect)

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


def draw_polygon_thick(draw, xy, fill=None, outline=None):
    draw.polygon(xy.flatten().tolist(), fill=fill, outline=outline)
    draw.polygon((xy-1).flatten().tolist(), fill=fill, outline=outline)
    draw.polygon((xy+1).flatten().tolist(), fill=fill, outline=outline)


def draw_one_entity_3dbbox_pillow(row, view_matrix, proj_matrix, width, height, im):
    bbox, bbox_2d = calculate_one_entity_bbox(row, view_matrix, proj_matrix, width, height)

    draw = ImageDraw.Draw(im)
    draw.rectangle(bbox[0, :].tolist() + bbox[1, :].tolist(), outline=colors.to_hex('y'))
    draw.rectangle((bbox[0, :] + 1).tolist() + (bbox[1, :] + 1).tolist(), outline=colors.to_hex('y'))   # increase width

    draw_polygon_thick(draw, bbox_2d[(0, 1, 3, 2), :], outline=colors.to_hex('c'))
    draw_polygon_thick(draw, bbox_2d[(4, 5, 7, 6), :], outline=colors.to_hex('c'))
    draw_polygon_thick(draw, bbox_2d[(0, 2, 6, 4), :], outline=colors.to_hex('c'))
    draw_polygon_thick(draw, bbox_2d[(1, 3, 7, 5), :], outline=colors.to_hex('c'))
    draw_polygon_thick(draw, bbox_2d[(0, 1, 5, 4), :], outline=colors.to_hex('r'))
    draw_polygon_thick(draw, bbox_2d[(2, 3, 7, 6), :], outline=colors.to_hex('g'))


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
    return np.copy(depths[name])    # it is being modified, now loaded data dont mutate


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
    return np.copy(stencils[name])


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


@lru_cache(maxsize=8)
def get_first_record_timestamp_in_run(run_id):
    conn = get_connection_pooled()
    cur = conn.cursor()
    cur.execute("""SELECT min(timestamp) as timestamp 
        FROM snapshots 
        WHERE run_id = {} 
        LIMIT 1 
        """.format(run_id))
    return cur.fetchone()['timestamp']


def is_first_record_in_run(res, run_id):
    first_timestamp = get_first_record_timestamp_in_run(run_id)
    return first_timestamp == res['timestamp']


def get_previous_record(res):
    conn = get_connection_pooled()
    cur = conn.cursor()
    cur.execute("""SELECT imagepath, snapshot_id, scene_id 
        FROM snapshots 
        WHERE timestamp < '{}' and run_id = (SELECT run_id from snapshots WHERE snapshot_id = {}) 
        ORDER BY timestamp DESC 
        LIMIT 1 
        """.format(res['timestamp'], res['snapshot_id']))
    # this should select previous record independently on primary key, without problems
    # with race conditions by persisting in other threads
    # and belonging into the same run
    results = []
    for row in cur:
        res = dict(row)
        results.append(res)
    if len(results) == 0:
        print('no previous record for snapshot_id {}'.format(res['snapshot_id']))
    return results[0]['imagepath']


def are_buffers_same_as_previous(res):
    name = res['imagepath']
    depth = load_depth(name)
    stencil = load_stencil(name)
    prev_name = get_previous_record(res)
    prev_depth = load_depth(prev_name)
    prev_stencil = load_stencil(prev_name)
    return (depth == prev_depth).all() or (stencil == prev_stencil).all()


def camera_to_string(res):
    return 'camera_{}__{}'.format(
        '_'.join(['{:0.2f}'.format(i) for i in res['camera_relative_position']]),
        '_'.join(['{:0.2f}'.format(i) for i in res['camera_relative_rotation']]),
    )


def get_dataset_filename_wildcard():
    return '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]--[0-9][0-9]-[0-9][0-9]-[0-9][0-9]--[0-9][0-9][0-9]'


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


def save_pointcloud_csv(vecs, name, paraview=False):
    if paraview:
        assert (vecs.shape[1] == 4)
    else:
        assert (vecs.shape[1] == 3)
    a = np.asarray(vecs)
    np.savetxt(name, a, delimiter=",")


if __name__ == '__main__':
    main()

# online combining to multipage tiff https://www.coolutils.com/Online/TIFF-Combine/
