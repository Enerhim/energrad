#include "include/context.cuh"
#include "include/engine.cuh"
#include "include/tensor.cuh"
#include <iostream>

void printTensor(Tensor t) {
  auto shape = t->getShape();
  auto strides = t->getStrides();
  size_t ndims = shape.size();
  size_t nelements = 1;
  for (auto s : shape) {
    nelements *= s;
  }

  std::vector<float> host_data = t->toHost();
  float *data = host_data.data();

  std::vector<int> indices(ndims, 0);

  for (int i = 0; i < nelements; i++) {
    int offset = 0;

    for (int k = 0; k < ndims; k++) {
      offset += strides[k] * indices[k];
    }

    std::cout << data[offset] << " ";

    indices[ndims - 1]++;
    for (int k = ndims - 1; k > 0; k--) {
      if (indices[k] >= shape[k]) {
        indices[k] = 0;
        indices[k - 1]++;
      } else {
        break;
      }
    }
  }

  std::cout << std::endl;
}

int main() {
  std::shared_ptr<CudaContext> ctx = std::make_shared<CudaContext>();
  Engine engine;

  Tensor A = tensor("", {2, 2}, {1, 2, 3, 4}, ctx, nullptr, true, false);
  Tensor B = tensor("", {2, 2}, {1, 2, 3, 4}, ctx, nullptr, true, false);
  auto C = A + B;
  printTensor(A);
  printTensor(B);
  printTensor(C);
  engine.backward(C.get());
  printTensor(A->getGradContext()->grad);
  printTensor(B->getGradContext()->grad);
  printTensor(C->getGradContext()->grad);
  return 0;
}
