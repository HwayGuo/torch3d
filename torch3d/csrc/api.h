// -*- mode: c++ -*-
#include <torch/extension.h>
#include <vector>

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")


at::Tensor farthest_point_sample(const at::Tensor& points, int num_samples);
at::Tensor ball_point(
    const at::Tensor& points,
    const at::Tensor& queries,
    int k,
    float radius);
std::vector<at::Tensor> voxelize_point_cloud(
    const at::Tensor& points,
    const at::Tensor& voxel_size,
    const at::Tensor& grid_size,
    const at::Tensor& min_range,
    const at::Tensor& max_range,
    int max_points_per_voxels,
    int max_voxels);
