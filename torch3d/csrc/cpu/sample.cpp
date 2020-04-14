#include <math.h>
#include "cpu.h"


template <typename T>
void farthest_point_sample_impl(
    const T* points,
    int batch_size,
    int num_points,
    int num_samples,
    int in_channels,
    T* sqdist,
    int64_t* indices)
{
    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t i = 0;

        for (int64_t m = 1; m < num_samples; ++m) {
            T maxval = 0;
            int64_t argmax = 0;
            T x = points[0 * num_points + i];
            T y = points[1 * num_points + i];
            T z = points[2 * num_points + i];

            for (int ii = 0; ii < num_points; ++ii) {
                T xx = points[0 * num_points + ii] - x;
                T yy = points[1 * num_points + ii] - y;
                T zz = points[2 * num_points + ii] - z;
                T d2 = xx * xx + yy * yy + zz * zz;
                sqdist[ii] = fmin(d2, sqdist[ii]);

                if (sqdist[ii] > maxval) {
                    argmax = ii;
                    maxval = d2;
                }
            }

            i = argmax;
            indices[m] = argmax;
        }

        points += in_channels * num_points;
        sqdist += num_points;
        indices += num_samples;
    }
}


at::Tensor farthest_point_sample_cpu(const at::Tensor& points, int num_samples)
{
    int batch_size = points.size(0);
    int in_channels = points.size(1);
    int num_points = points.size(2);
    at::Tensor indices =
        at::zeros({batch_size, num_samples}, points.options().dtype(at::kLong));
    at::Tensor sqdist = at::zeros({batch_size, num_points}, points.options()).fill_(1e10);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "farthest_point_sample_cpu", [&] {
        farthest_point_sample_impl<scalar_t>(
            points.data_ptr<scalar_t>(),
            batch_size,
            num_points,
            num_samples,
            in_channels,
            sqdist.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>());
    });

    return indices;
}
