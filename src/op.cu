#include "../include/op.cuh"

// AddOp

void AddOp::backward(float *top_gradient) {
  auto a = parents[0].lock();
  auto b = parents[1].lock();

  a->accumulateGrad(top_gradient);
  b->accumulateGrad(top_gradient);
}

// MatmulOp
