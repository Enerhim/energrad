#include "include/context.cuh"
#include "include/engine.cuh"
#include "include/op.cuh"
#include "include/tensor.cuh"
#include <iostream>

void printTensor(Tensor t) {
  if (!t) {
    std::cout << "None" << std::endl;
    return;
  }

  std::vector<float> host_data = t->toHost();
  for (auto e : host_data) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
}

int main() {
  std::shared_ptr<CudaContext> ctx = std::make_shared<CudaContext>();
  Engine engine;

  Tensor A = tensor("", {2, 2}, {1, 2, 3, 4}, ctx, nullptr, true, false);
  auto B = transpose(A, 0, 1);
  printTensor(A);
  printTensor(B);
  engine.backward(B.get());
  printTensor(A->getGradContext()->grad);
  printTensor(B->getGradContext()->grad);

  if (A->checkContiguous())
    std::cout << "Contiguous A!" << std::endl;
  if (B->checkContiguous())
    std::cout << "Contiguous B!" << std::endl;
  if (A->getGradContext()->grad->checkContiguous())
    std::cout << "Contiguous A grad!" << std::endl;
  if (B->getGradContext()->grad->checkContiguous())
    std::cout << "Contiguous B grad!" << std::endl;
  return 0;
}
