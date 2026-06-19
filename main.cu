#include "include/cuda_context.cuh"
#include "include/engine.cuh"
#include "include/tensor.cuh"
#include <iostream>

int main() {
  std::shared_ptr<CudaContext> ctx = std::make_shared<CudaContext>();
  Engine engine;

  Tensor A = make_tensor("A", {2, 2}, true, {1, 2, 3, 4}, ctx);
  Tensor B = make_tensor("B", {2, 2}, true, {1, 2, 3, 4}, ctx);

  Tensor C = A + B;
  cudaStreamSynchronize(ctx->stream);

  engine.backward(C);

  for (float x : A->hostGradBuffer()) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  for (float x : B->hostGradBuffer()) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  for (float x : C->hostGradBuffer()) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  return 0;
}
