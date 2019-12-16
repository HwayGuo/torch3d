#include "cuda.h"


template <typename T>
__global__ void ball_point_kernel(
    const T* p,
    const T* q,
    int B,
    int N,
    int M,
    int C,
    int K,
    float radius,
    int64_t* __restrict__ index)
{
    int b = blockIdx.x;

    p += b * N * C;
    q += b * M * C;
    index += b * M * K;

    int tid = threadIdx.x;
    float r2 = radius * radius;

    for (int64_t i = tid; i < M; i += NUM_THREADS) {
        int64_t k = 0;
        T x = q[i * C + 0];
        T y = q[i * C + 1];
        T z = q[i * C + 2];

        for (int64_t ii = 0; ii < N && k < K; ++ii) {
            T xx = p[ii * C + 0] - x;
            T yy = p[ii * C + 1] - y;
            T zz = p[ii * C + 2] - z;
            T d2 = xx * xx + yy * yy + zz * zz;

            if (d2 < r2) {
                if (k == 0) {
                    for (int l = 0; l < K; ++l)
                        index[i * K + l] = ii;
                }
                index[i * K + k] = ii;
                ++k;
            }
        }
    }
}


at::Tensor ball_point_cuda(const at::Tensor& p, const at::Tensor& q, int K, float radius)
{
    int B = p.size(0);
    int N = p.size(1);
    int M = q.size(1);
    int C = p.size(2);
    at::Tensor index = at::zeros({B, M, K}, p.options().dtype(at::kLong));

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "ball_point_cuda", [&] {
        dim3 block(NUM_THREADS);
        dim3 grid(B);
        ball_point_kernel<scalar_t><<<grid, block>>>(
            p.data_ptr<scalar_t>(),
            q.data_ptr<scalar_t>(),
            B,
            N,
            M,
            C,
            K,
            radius,
            index.data<int64_t>());
    });
    CUDA_CHECK_ERRORS();

    return index;
}
