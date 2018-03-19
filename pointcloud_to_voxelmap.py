import numpy as np
from voxel_map import VoxelMap
import pickle


def pointcloud_to_voxelmap(pointcloud):
    map = VoxelMap()
    map.voxel_size = 1
    map.free_update = -1.0
    map.hit_update = 1.0
    map.occupancy_threshold = 0.0
    print([method_name for method_name in dir(VoxelMap) if callable(getattr(VoxelMap, method_name))])


def pointcloud_from_csv(path):
    return np.loadtxt(path, dtype=np.float32, delimiter=',')


def playing_with_voxelmap():
    map = VoxelMap()
    map.voxel_size = 1
    map.free_update = -1.0
    map.hit_update = 1.0
    map.occupancy_threshold = 0.5

    x0 = np.array([
        [1, 1],
        [1, 1],
        [1, 1],
    ], dtype=np.float32)
    x1 = np.array([
        [4, 5],
        [4, 3],
        [2, 1],
    ], dtype=np.float32)

    map.update_lines(x0, x1)

    [voxels, levels, values] = map.get_voxels()
    # ret = map.trace_lines()
    with open('voxels.rick', 'wb+') as f:
        pickle.dump([voxels, values], f)


if __name__ == '__main__':
    playing_with_voxelmap()

    point_cloud = pointcloud_from_csv('../GTAVisionExport_postprocessing/points-2018-02-16--17-09-02--180.csv')
    pointcloud_to_voxelmap(point_cloud.T)
