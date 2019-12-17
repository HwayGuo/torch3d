#include <math.h>
#include "cpu.h"


template <typename T>
void farthest_point_sample_impl(
    const T* p,
    int B,
    int N,
    int M,
    int C,
    T* sqdist,
    int64_t* index)
{
    for (int64_t b = 0; b < B; ++b) {
        int64_t i = 0;

        for (int64_t m = 1; m < M; ++m) {
            T maxval = 0;
            int64_t argmax = 0;
            T x = p[0 * N + i];
            T y = p[1 * N + i];
            T z = p[2 * N + i];

            for (int ii = 0; ii < N; ++ii) {
                T xx = p[0 * N + ii] - x;
                T yy = p[1 * N + ii] - y;
                T zz = p[2 * N + ii] - z;
                T d2 = xx * xx + yy * yy + zz * zz;
                sqdist[ii] = fmin(d2, sqdist[ii]);

                if (sqdist[ii] > maxval) {
                    argmax = ii;
                    maxval = d2;
                }
            }

            i = argmax;
            index[m] = argmax;
        }

        p += C * N;
        sqdist += N;
        index += M;
    }
}


at::Tensor farthest_point_sample_cpu(const at::Tensor& p, int M)
{
    int B = p.size(0);
    int N = p.size(2);
    int C = p.size(1);
    at::Tensor index = at::zeros({B, M}, p.options().dtype(at::kLong));
    at::Tensor sqdist = at::zeros({B, N}, p.options()).fill_(1e10);

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "farthest_point_sample_cpu", [&] {
        farthest_point_sample_impl<scalar_t>(
            p.data_ptr<scalar_t>(),
            B,
            N,
            M,
            C,
            sqdist.data_ptr<scalar_t>(),
            index.data_ptr<int64_t>());
    });

    return index;
}
