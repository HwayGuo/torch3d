import torch


def voxelize_point_cloud(
    points,
    voxel_size,
    grid_size,
    min_range,
    max_range,
    max_points_per_voxels=128,
    max_voxels=20000,
):
    if grid_size is None:
        grid_size = torch.floor((max_range - min_range) / voxel_size)
    _C = _lazy_import()
    voxels, indices, num_points_per_voxel = _C.voxelize_point_cloud(
        points,
        voxel_size,
        grid_size,
        min_range,
        max_range,
        max_points_per_voxels,
        max_voxels,
    )
    return voxels, indices, num_points_per_voxel
