// -*- mode: c++ -*-
#include <torch/types.h>

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

#define CUDA_CHECK_ERRORS()                                    \
    do {                                                       \
        cudaError_t err = cudaGetLastError();                  \
        if (cudaSuccess != err) {                              \
            fprintf(                                           \
                stderr,                                        \
                "CUDA kernel failed : %s\n%s at L:%d in %s\n", \
                cudaGetErrorString(err),                       \
                __PRETTY_FUNCTION__,                           \
                __LINE__,                                      \
                __FILE__);                                     \
            exit(-1);                                          \
        }                                                      \
    } while (0)

#define NUM_THREADS 256


at::Tensor farthest_point_sample_cuda(const at::Tensor& p, int num_samples);
at::Tensor ball_point_cuda(const at::Tensor& p, const at::Tensor& q, int k, float radius);
