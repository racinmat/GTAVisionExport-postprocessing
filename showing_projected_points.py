import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

if __name__ == '__main__':
    with open('points.rick', mode='rb') as file:
        struct = pickle.load(file)

    points = struct['points']
    vecs_p = struct['vecs_p']
    colors = struct['colors']
    name = struct['name']

    # transformation to pointcloud form
    xs = vecs_p[0, :]
    ys = vecs_p[1, :]
    zs = vecs_p[2, :]

    # visualization
    fig = plt.figure()

    ax = Axes3D(fig)
    #ax = fig.add_subplot(111, projection='3d')
    #for i in range(vecs_p.shape[1]):
    #    ax.scatter(xs[i], ys[i], zs[i], c=colors[i], marker='o')
    ax.scatter(xs, ys, zs, c='b', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
