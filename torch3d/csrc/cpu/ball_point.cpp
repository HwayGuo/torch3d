#include "cpu.h"


template <typename T>
void ball_point_impl(
    const T* p,
    const T* q,
    int B,
    int N,
    int M,
    int C,
    int K,
    float radius,
    int64_t* index)
{
    const float r2 = radius * radius;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t i = 0; i < M; ++i) {
            T x = q[0 * M + i];
            T y = q[1 * M + i];
            T z = q[2 * M + i];

            int64_t ii;
            int64_t k = 0;
            for (ii = 0; ii < N && k < K; ++ii) {
                T xx = p[0 * N + ii] - x;
                T yy = p[1 * N + ii] - y;
                T zz = p[2 * N + ii] - z;
                T d2 = xx * xx + yy * yy + zz * zz;

                if (d2 < r2) {
                    if (k == 0) {
                        for (int l = 0; l < K; ++l)
                            index[l * M + i] = ii;
                    }
                    index[k * M + i] = ii;
                    k++;
                }
            }
        }

        p += C * N;
        q += C * M;
        index += K * M;
    }
}


at::Tensor ball_point_cpu(const at::Tensor& p, const at::Tensor& q, int K, float radius)
{
    int B = p.size(0);
    int N = p.size(2);
    int M = q.size(2);
    int C = p.size(1);
    at::Tensor index = at::zeros({B, K, M}, p.options().dtype(at::kLong));

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "ball_point_cpu", [&] {
        ball_point_impl<scalar_t>(
            p.data_ptr<scalar_t>(),
            q.data_ptr<scalar_t>(),
            B,
            N,
            M,
            C,
            K,
            radius,
            index.data_ptr<int64_t>());
    });

    return index;
}
