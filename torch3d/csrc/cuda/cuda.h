// -*- mode: c++ -*-
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val == 0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


at::Tensor farthest_point_sample_cuda(const at::Tensor& input, int m);

at::Tensor ball_point_cuda(
    const at::Tensor& input,
    const at::Tensor& query,
    float radius,
    int k);

at::Tensor point_interpolate_cuda(
    const at::Tensor& input,
    const at::Tensor& index,
    const at::Tensor& weight);
at::Tensor point_interpolate_grad_cuda(
    const at::Tensor& grad,
    const at::Tensor& index,
    const at::Tensor& weight,
    int n);
