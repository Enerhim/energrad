#include "include/context.cuh"
#include "include/engine.cuh"
#include "include/tensor.cuh"
#include <iostream>

void printTensor(Tensor t) {
  std::vector<float> host_data = t->toHost();
  for (auto e : host_data) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
}

// TODO: 0D Tensor Support

int main() {
  std::shared_ptr<CudaContext> ctx = std::make_shared<CudaContext>();
  Engine engine;

  Tensor A = tensor("", {2, 2}, {1, 2, 3, 4}, ctx, nullptr, true, false);
  Tensor B = tensor("", {1, 1}, {4}, ctx, nullptr, true, false);
  auto C = A + expand(B, A->getShape());
  printTensor(A);
  printTensor(B);
  printTensor(C);
  return 0;
}
