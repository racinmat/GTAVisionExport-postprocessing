import scipy.io

from stl import mesh
import math
import numpy as np
# Optionally render the rotated cube faces
from matplotlib import pyplot
from mpl_toolkits import mplot3d

if __name__ == '__main__':
    # Create tetrahedron
    data = np.zeros(6, dtype=mesh.Mesh.dtype)

    # tetrahedron base 1
    data['vectors'][0] = np.array(
        [[1, 1, -1],
         [1, -1, -1],
         [-1, -1, -1]])
    # tetrahedron base 1
    data['vectors'][1] = np.array(
        [[1, 1, -1],
         [-1, 1, -1],
         [-1, -1, -1]])
    # sides
    data['vectors'][2] = np.array(
        [[1, 1, -1],
         [-1, 1, -1],
         [0, 0, 1]])
    data['vectors'][3] = np.array(
        [[1, 1, -1],
         [1, -1, -1],
         [0, 0, 1]])
    data['vectors'][4] = np.array(
        [[-1, -1, -1],
         [-1, 1, -1],
         [0, 0, 1]])
    data['vectors'][5] = np.array(
        [[-1, -1, -1],
         [1, -1, -1],
         [0, 0, 1]])

    m = mesh.Mesh(data.copy())
    # Create a new plot
    # figure = pyplot.figure()
    # axes = mplot3d.Axes3D(figure)
    #
    # # Render the cube faces
    # axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
    #
    # # Auto scale to the mesh size
    # scale = np.concatenate(m.points).flatten(-1)
    # axes.auto_scale_xyz(scale, scale, scale)
    #
    # # Show the plot to the screen
    # pyplot.show()

    m.save('../data-pipeline-demonstration/camera.stl')

    depth_img = scipy.io.loadmat(r'C:\Users\Azathoth\Downloads\Train400Depth\Train400Depth\depth_sph_corr-op1-p-108t000.mat')
    depth_img = depth_img['Position3DGrid']
    print(depth_img[:,:,3].shape)
    # depth_img = depth_img.astype(dtype=np.uint8)
    pyplot.figure()
    pyplot.imshow(depth_img[:,:,0:3].astype(np.uint8))
    # pyplot.imshow(depth_img[:,:,3])
    pyplot.show()