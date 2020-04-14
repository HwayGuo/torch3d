// -*- mode: c++ -*-
#include <torch/types.h>
#include <vector>


at::Tensor farthest_point_sample_cpu(const at::Tensor& points, int num_samples);
at::Tensor ball_point_cpu(
    const at::Tensor& points,
    const at::Tensor& queries,
    int k,
    float radius);
std::vector<at::Tensor> voxelize_point_cloud_cpu(
    const at::Tensor& points,
    const at::Tensor& voxel_size,
    const at::Tensor& grid_size,
    const at::Tensor& min_range,
    const at::Tensor& max_range,
    int max_points_per_voxel,
    int max_voxels);
