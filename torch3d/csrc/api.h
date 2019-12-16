// -*- mode: c++ -*-
#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")


at::Tensor farthest_point_sample(const at::Tensor& p, int num_samples);
at::Tensor ball_point(const at::Tensor& p, const at::Tensor& q, int k, float radius);

at::Tensor point_interpolate(
    const at::Tensor& input,
    const at::Tensor& index,
    const at::Tensor& weight);
at::Tensor point_interpolate_grad(
    const at::Tensor& grad,
    const at::Tensor& index,
    const at::Tensor& weight,
    int n);
