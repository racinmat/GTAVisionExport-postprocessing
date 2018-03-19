import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
import numpy as np

def cube(ax, center, l, opacity):  # plots a cube of side l at (a,b,c)
    a, b, c = center
    for ll in [0, l]:
        for i in range(3):
            dire = ["x", "y", "z"]
            xdire = [b, a, a]
            ydire = [c, c, b]
            zdire = [a, b, c]
            # ax.add_collection3d(Poly3DCollection(verts))
            side = Rectangle((xdire[i], ydire[i]), l, l, edgecolor=None, color='red')
            ax.add_patch(side)
            art3d.pathpatch_2d_to_3d(side, z=zdire[i] + ll, zdir=dire[i])


def show_voxels(voxels, values):
    voxel_size = 1.0

    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")

    # x, y, z = np.indices((8, 8, 8))
    mins = np.min(voxels, axis=1)
    maxs = np.max(voxels, axis=1)
    diffs = (maxs - mins) + 1
    ranges = (diffs * voxel_size).astype(int)
    # x, y, z = np.indices(ranges) / voxel_size + voxel_size / 2
    x, y, z = np.indices(ranges) / voxel_size
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
    xc, yc, zc = np.indices(ranges + 1) / voxel_size
    xc += min_corners[0]
    yc += min_corners[1]
    zc += min_corners[2]

    ax.voxels(xc, yc, zc, filled, facecolors=all_values, edgecolor='k')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def example_voxels():

    def midpoints(x):
        sl = ()
        for i in range(x.ndim):
            x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
            sl += np.index_exp[:]
        return x

    # prepare some coordinates, and attach rgb values to each
    r, g, b = np.indices((17, 17, 17)) / 16.0
    rc = midpoints(r)
    gc = midpoints(g)
    bc = midpoints(b)

    # define a sphere about [0.5, 0.5, 0.5]
    sphere = (rc - 0.5) ** 2 + (gc - 0.5) ** 2 + (bc - 0.5) ** 2 < 0.5 ** 2

    # combine the color components
    colors = np.zeros(sphere.shape + (3,))
    colors[..., 0] = rc
    colors[..., 1] = gc
    colors[..., 2] = bc

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(r, g, b, sphere,
              facecolors=colors,
              edgecolors=np.clip(2 * colors - 0.5, 0, 1),  # brighter
              linewidth=0.5)
    ax.set(xlabel='r', ylabel='g', zlabel='b')

    plt.show()

if __name__ == '__main__':
    with open('voxels.rick', 'rb') as f:
        voxels, values = pickle.load(f)
    example_voxels()
    show_voxels(voxels, values)