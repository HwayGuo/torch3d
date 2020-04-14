#include "cpu.h"


template <typename T>
void voxelize_point_cloud_impl(const T* points)
{
}


std::vector<at::Tensor> voxelize_point_cloud_cpu(
    const at::Tensor& points,
    const at::Tensor& voxel_size,
    const at::Tensor& grid_size,
    const at::Tensor& min_range,
    const at::Tensor& max_range,
    int max_points_per_voxel,
    int max_voxels)
{
    int batch_size = points.size(0);
    int in_channels = points.size(1);
    int num_points = points.size(2);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "voxelize_point_cloud_cpu", [&] {
        voxelize_point_cloud_impl<scalar_t>(points.data_ptr<scalar_t>());
    });
}
