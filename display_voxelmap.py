import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
import numpy as np


def show_voxels_3d(voxels, values, voxel_size):

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
    # ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax = plt.subplot(gs[0], projection='3d')
    # ax.set_aspect("equal")

    mins = np.min(voxels, axis=1)
    maxs = np.max(voxels, axis=1)
    diffs = (maxs - mins) / voxel_size
    ranges = (diffs + 1).astype(int)
    # x, y, z = np.indices(ranges) / voxel_size + voxel_size / 2
    x, y, z = np.indices(ranges) * voxel_size
    x += mins[0]
    y += mins[1]
    z += mins[2]
    all_voxels = np.array((x, y, z))
    all_voxels_flat = all_voxels.reshape(3, -1) # this way I obtain list of all voxels
    filled = np.array([(row == voxels.T).all(1).any() for row in all_voxels_flat.T]) # now I compare these voxels with present voxels, obtaining array of booleans
    all_values = np.zeros_like(filled, dtype=np.float32)
    all_values[filled] = values
    filled = filled.reshape(x.shape) # reshaping array of booleans back to topological representation
    all_values = all_values.reshape(x.shape) # reshaping array of booleans back to topological representation

    # x, y, z are supposed to be corners of voxels, so they will be 1 dim bigger and moved, because I have centers
    min_corners = mins - voxel_size / 2
    xc, yc, zc = np.indices(ranges + 1) * voxel_size
    xc += min_corners[0]
    yc += min_corners[1]
    zc += min_corners[2]

    values_range = (np.max(all_values) - np.min(all_values))
    values_min = np.min(all_values)
    # because I must deal with colormap mapping myself, fuck it
    cmap_keys = (all_values - values_min) / values_range # remapping values to 0 1 range
    colors = plt.cm.get_cmap('plasma')(cmap_keys)
    # all_values += np.min(all_values) # colormaps don't like negative values, shifting by offset

    ax.voxels(xc, yc, zc, filled, facecolors=colors, edgecolor='k')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    # showing colormap just for seeing the values
    # ax = fig.add_subplot(2, 1, 2)
    ax = plt.subplot(gs[1])

    gradient = np.linspace(0, 1, 256)
    gradients = np.vstack((gradient, gradient))
    ax.imshow(gradients.T, aspect='auto', cmap=plt.get_cmap('plasma'))
    positions = list(range(len(gradient)))
    labels = np.round((gradient * values_range) + values_min, decimals=2)
    plt.yticks(positions[::10], labels[::10])
    plt.show()


def show_voxels_as_points(voxels, values, voxel_size):
    fig = plt.figure()

    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
    # ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax = plt.subplot(gs[0], projection='3d')
    xs = voxels[0, :]
    ys = voxels[1, :]
    zs = voxels[2, :]

    # x, y, z are supposed to be corners of voxels, so they will be 1 dim bigger and moved, because I have centers
    values_range = (np.max(values) - np.min(values))
    values_min = np.min(values)
    # because I must deal with colormap mapping myself, fuck it
    cmap_keys = (values - values_min) / values_range # remapping values to 0 1 range
    colors = plt.cm.get_cmap('plasma')(cmap_keys)

    ax.scatter(xs, ys, zs, c=colors, marker='x')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    # showing colormap just for seeing the values
    # ax = fig.add_subplot(2, 1, 2)
    ax = plt.subplot(gs[1])

    gradient = np.linspace(0, 1, 256)
    gradients = np.vstack((gradient, gradient))
    ax.imshow(gradients.T, aspect='auto', cmap=plt.get_cmap('plasma'))
    positions = list(range(len(gradient)))
    labels = np.round((gradient * values_range) + values_min, decimals=2)
    plt.yticks(positions[::10], labels[::10])
    plt.show()


def save_csv(voxels, name):
    a = np.asarray(voxels)
    np.savetxt("voxels-{}.csv".format(name), a, delimiter=",")


if __name__ == '__main__':
    name = 'orig-short-2018-03-07--18-26-53--512'
    with open('voxelmap-{}.rick'.format(name), 'rb') as f:
        voxels, values, voxel_size = pickle.load(f)
    # show_voxels_3d(voxels, values, voxel_size)
    voxels = voxels[:, values >= 0]
    values = values[values >= 0]
    # show_voxels_as_points(voxels, values, voxel_size)
    save_csv(voxels.T, name)
