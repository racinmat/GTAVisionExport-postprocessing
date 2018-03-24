import numpy as np
from voxel_map import VoxelMap
import pickle


def pointcloud_to_voxelmap(pointcloud, camera_pos=np.array([0, 0, 0]), voxel_size=0.25, free_update=-1.0, hit_update=2.0):
    map = VoxelMap()
    map.voxel_size = voxel_size
    map.free_update = free_update
    map.hit_update = hit_update  # zkusit 2násobný hit oproti free, zkusit include directories env var
    map.occupancy_threshold = 0.0
    line_starts = np.repeat(camera_pos[:, np.newaxis], pointcloud.shape[1], axis=1)
    map.update_lines(line_starts, pointcloud)
    [voxels, levels, values] = map.get_voxels()
    # size je počet známých voxelů, počet prvků v hashmapě
    # můžu získávat i hodnoty konkrétních voxelů přes map.get_voxels(voxels, levels)
    # voxely zobrazovat jako pointcloud
    return voxels, values, map.voxel_size


def pointclouds_to_voxelmap(pointclouds, camera_posisions, voxel_size=0.25, free_update=-1.0, hit_update=2.0):
    assert len(pointclouds) == len(camera_posisions)

    map = VoxelMap()
    map.voxel_size = voxel_size
    map.free_update = free_update
    map.hit_update = hit_update  # zkusit 2násobný hit oproti free, zkusit include directories env var
    map.occupancy_threshold = 0.0
    for pointcloud, cam_pos in zip(pointclouds, camera_posisions):
        line_starts = np.repeat(cam_pos[:, np.newaxis], pointcloud.shape[1], axis=1)
        map.update_lines(line_starts, pointcloud)
    [voxels, levels, values] = map.get_voxels()
    # size je počet známých voxelů, počet prvků v hashmapě
    # můžu získávat i hodnoty konkrétních voxelů přes map.get_voxels(voxels, levels)
    # voxely zobrazovat jako pointcloud
    with open('voxelmap-{}.rick'.format(name), 'wb+') as f:
        pickle.dump([voxels, values, map.voxel_size], f)


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
    # playing_with_voxelmap()

    name = 'orig-short-2018-03-07--18-26-53--512'
    point_cloud = pointcloud_from_csv('../GTAVisionExport_postprocessing/points-{}.csv'.format(name))
    voxels, values, voxel_size = pointcloud_to_voxelmap(point_cloud.T)
    with open('voxelmap-{}.rick'.format(name), 'wb+') as f:
        pickle.dump([voxels, values, voxel_size], f)
