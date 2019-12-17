#include "cuda.h"


template <typename T>
__device__ void __update(
    T* __restrict__ sdist,
    int64_t* __restrict__ sdist_i,
    int64_t i,
    int64_t j)
{
    const T v1 = sdist[i];
    const T v2 = sdist[j];
    const int64_t i1 = sdist_i[i];
    const int64_t i2 = sdist_i[j];
    sdist[i] = max(v1, v2);
    sdist_i[i] = v2 > v1 ? i2 : i1;
}


template <typename T>
__global__ void farthest_point_sample_kernel(
    const T* __restrict__ p,
    int B,
    int N,
    int M,
    int C,
    T* __restrict__ sqdist,
    int64_t* __restrict__ index)
{
    __shared__ T sdist[NUM_THREADS];
    __shared__ int64_t sdist_i[NUM_THREADS];

    int b = blockIdx.x;

    p += b * C * N;
    sqdist += b * N;
    index += b * M;

    int64_t i = 0;
    int tid = threadIdx.x;

    for (int64_t m = 1; m < M; ++m) {
        T maxval = 0;
        int64_t argmax = 0;
        T x = p[0 * N + i];
        T y = p[1 * N + i];
        T z = p[2 * N + i];

        for (int ii = tid; ii < N; ii += NUM_THREADS) {
            T xx = p[0 * N + ii] - x;
            T yy = p[1 * N + ii] - y;
            T zz = p[2 * N + ii] - z;
            T d = xx * xx + yy * yy + zz * zz;
            d = min(d, sqdist[ii]);
            sqdist[ii] = d;
            if (d > maxval) {
                argmax = ii;
                maxval = d;
            }
        }
        sdist[tid] = maxval;
        sdist_i[tid] = argmax;
        __syncthreads();

        if (NUM_THREADS >= 512) {
            if (tid < 256) {
                __update(sdist, sdist_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (NUM_THREADS >= 256) {
            if (tid < 128) {
                __update(sdist, sdist_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (NUM_THREADS >= 128) {
            if (tid < 64) {
                __update(sdist, sdist_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (NUM_THREADS >= 64) {
            if (tid < 32) {
                __update(sdist, sdist_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (NUM_THREADS >= 32) {
            if (tid < 16) {
                __update(sdist, sdist_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (NUM_THREADS >= 16) {
            if (tid < 8) {
                __update(sdist, sdist_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (NUM_THREADS >= 8) {
            if (tid < 4) {
                __update(sdist, sdist_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (NUM_THREADS >= 4) {
            if (tid < 2) {
                __update(sdist, sdist_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (NUM_THREADS >= 2) {
            if (tid < 1) {
                __update(sdist, sdist_i, tid, tid + 1);
            }
            __syncthreads();
        }

        i = sdist_i[0];
        if (tid == 0)
            index[m] = i;
    }
}


at::Tensor farthest_point_sample_cuda(const at::Tensor& p, int M)
{
    int B = p.size(0);
    int N = p.size(2);
    int C = p.size(1);
    at::Tensor index = at::zeros({B, M}, p.options().dtype(at::kLong));
    at::Tensor sqdist = at::zeros({B, N}, p.options()).fill_(1e10);

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "farthest_point_sample_cuda", [&] {
        dim3 block(NUM_THREADS);
        dim3 grid(B);
        farthest_point_sample_kernel<scalar_t><<<grid, block>>>(
            p.data_ptr<scalar_t>(),
            B,
            N,
            M,
            C,
            sqdist.data_ptr<scalar_t>(),
            index.data_ptr<int64_t>());
    });
    CUDA_CHECK_ERRORS();

    return index;
}
