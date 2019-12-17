// -*- mode: c++ -*-
#include <torch/types.h>


at::Tensor farthest_point_sample_cpu(const at::Tensor& p, int num_samples);
at::Tensor ball_point_cpu(const at::Tensor& p, const at::Tensor& q, int k, float radius);
