// -*- mode: c++ -*-
#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")


at::Tensor farthest_point_sample(const at::Tensor& p, int num_samples);
at::Tensor ball_point(const at::Tensor& p, const at::Tensor& q, int k, float radius);
