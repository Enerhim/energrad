#include "../include/op.cuh"

void AddOp::backward(const Buffer &top_gradient) {
  if (top_gradient.kind != MemoryKind::Device)
    throw std::runtime_error("Buffer kind for AddOp is not device.\n");

  auto a = parents[0].lock();
  auto b = parents[1].lock();

  a->accumulateGrad(top_gradient);
  b->accumulateGrad(top_gradient);
}
