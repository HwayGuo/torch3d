#include "api.h"
#include "cpu/cpu.h"
#ifdef WITH_CUDA
#include "cuda/cuda.h"
#endif


at::Tensor farthest_point_sample(const at::Tensor& p, int num_samples)
{
    CHECK_CONTIGUOUS(p);

    if (p.type().is_cuda()) {
#ifdef WITH_CUDA
        return farthest_point_sample_cuda(p, num_samples);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return farthest_point_sample_cpu(p, num_samples);
}


at::Tensor ball_point(const at::Tensor& p, const at::Tensor& q, int k, float radius)
{
    CHECK_CONTIGUOUS(p);
    CHECK_CONTIGUOUS(q);

    if (p.type().is_cuda()) {
#ifdef WITH_CUDA
        return ball_point_cuda(p, q, k, radius);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not compiled with GPU support");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("farthest_point_sample", &farthest_point_sample);
    m.def("ball_point", &ball_point);
}
