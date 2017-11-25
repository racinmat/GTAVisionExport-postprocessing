import os
import numpy as np
import re
from PIL import Image, ImageFile
from skimage import io
from matplotlib import cm, patches
import matplotlib.pyplot as plt
import psycopg2
import tifffile


def get_connection():
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='postgres'")
    return conn


def bbox_from_string(string):
    return np.array([float(i) for i in re.sub('[()]', '', string).split(',')]).reshape(2, 2)


def show_bounding_boxes(name, size, ax):
    name = name.replace('info-', '')
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""SELECT bbox, 
        ARRAY[st_xmin(bbox3d), st_xmax(bbox3d), st_ymin(bbox3d), st_ymax(bbox3d), st_zmin(bbox3d), st_zmax(bbox3d)], 
        view_matrix, proj_matrix
        FROM detections
        JOIN snapshots ON detections.snapshot_id = snapshots.snapshot_id
        WHERE imagepath = '{}'
        AND NOT bbox @> POINT '(Infinity, Infinity)'""".format(name))
    rows = cur.fetchall()
    print(size)
    for row in rows:
        # bbox format is
        # [max x, max y]
        # [min x, min y]
        bbox = bbox_from_string(row[0])
        print(bbox)
        # bbox_x = bbox[:,0]
        # bbox_y = bbox[:,1]
        bbox[:, 0] *= size[1]
        bbox[:, 1] *= size[0]
        print(bbox)
        bbox3d = np.array(row[1])
        view_matrix = np.array(row[2])
        proj_matrix = np.array(row[3])

        width, height = bbox[0, :] - bbox[1, :]
        rect = patches.Rectangle(bbox[1, :], width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    # print(rows)
    bbox = np.array([1, 1, 0, 0]).reshape(2, 2)
    # bbox_y = bbox[:,0]
    # bbox_x = bbox[:,1]
    bbox[:, 0] *= size[0]
    bbox[:, 1] *= size[1]

    height, width = bbox[0, :] - bbox[1, :]
    rect = patches.Rectangle(bbox[1, :], width, height, linewidth=3, edgecolor='y', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)


def load_depth(name):
    if name not in depths:
        tiff_depth = tifffile.imread(os.path.join(in_directory, name + '-depth.tiff'))
        depths[name] = tiff_depth
    return depths[name]


def load_stencil(name):
    if name not in stencils:
        tiff_stencil = tifffile.imread(os.path.join(in_directory, name + '-stencil.tiff'))
        stencils[name] = tiff_stencil
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
        'info-2017-11-19--23-21-08',
        'info-2017-11-19--23-20-45',
        'info-2017-11-19--23-20-49'
    ]
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for name in files:
        im = Image.open(os.path.join(in_directory, name + '.tiff'))
        size = (im.size[1], im.size[0])

        fig = plt.figure()
        plt.imshow(im)
        show_bounding_boxes(name, size, plt.gca())
        plt.savefig(os.path.join(out_directory, 'bboxes-' + name + '.jpg'))

        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # plt.tight_layout()
        #
        # plt.axis('off')
        # im.seek(0)
        # ax1.imshow(im)
        # # ax2.set_title('f')
        # ax2.imshow(ids_to_greyscale(load_stencil_ids(name)), cmap='gray')
        # ax3.set_title('ids')
        # # ax3.imshow(load_stencil_ids(name), cmap='gray')
        # ax3.imshow(ids_to_greyscale(load_stencil_ids(name)), cmap='plasma')
        # ax4.set_title('depth')
        # ax4.imshow(load_depth(name), cmap='gray')
        # show_bounding_boxes(name, size, ax1)
        # show_bounding_boxes(name, size, ax3)
        # plt.axis('off')
        # # plt.draw()
        # plt.show()

    plt.show()

depths = {}
stencils = {}
in_directory = './../output'
out_directory = './img'
if __name__ == '__main__':
    main()
