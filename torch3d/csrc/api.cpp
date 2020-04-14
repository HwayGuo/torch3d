#include "api.h"
#include "cpu/cpu.h"
#ifdef WITH_CUDA
#include "cuda/cuda.h"
#endif


at::Tensor farthest_point_sample(const at::Tensor& points, int num_samples)
{
    CHECK_CONTIGUOUS(points);

    if (points.type().is_cuda()) {
#ifdef WITH_CUDA
        return farthest_point_sample_cuda(points, num_samples);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return farthest_point_sample_cpu(points, num_samples);
}


at::Tensor ball_point(
    const at::Tensor& points,
    const at::Tensor& queries,
    int k,
    float radius)
{
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(queries);

    if (points.type().is_cuda()) {
#ifdef WITH_CUDA
        return ball_point_cuda(points, queries, k, radius);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return ball_point_cpu(points, queries, k, radius);
}


std::vector<at::Tensor> voxelize_point_cloud(
    const at::Tensor& points,
    const at::Tensor& voxel_size,
    const at::Tensor& grid_size,
    const at::Tensor& min_range,
    const at::Tensor& max_range,
    int max_points_per_voxels,
    int max_voxels)
{
    CHECK_CONTIGUOUS(points);

    if (points.type().is_cuda()) {
#ifdef WITH_CUDA
        return voxelize_point_cloud_cuda(
            points,
            voxel_size,
            grid_size,
            min_range,
            max_range,
            max_points_per_voxels,
            max_voxels);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return voxelize_point_cloud_cpu(
        points,
        voxel_size,
        grid_size,
        min_range,
        max_range,
        max_points_per_voxels,
        max_voxels);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("farthest_point_sample", &farthest_point_sample);
    m.def("ball_point", &ball_point);
    m.def("voxelize_point_cloud", &voxelize_point_cloud);
}
