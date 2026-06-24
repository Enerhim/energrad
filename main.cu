#include "include/cuda_context.cuh"
#include "include/engine.cuh"
#include "include/tensor.cuh"
#include <iostream>

void printTensor(const float *data, const std::vector<size_t> &shape,
                 const std::vector<size_t> &strides) {
  size_t ndims = shape.size();
  std::vector<int> indices(ndims, 0);

  size_t no_elements = 1;
  for (size_t s : shape) {
    no_elements *= s;
  }
  for (int i = 0; i < no_elements; i++) {
    int offset = 0;
    for (int k = 0; k < ndims; k++) {
      offset += strides[k] * indices[k];
    }
    std::cout << data[offset] << " ";

    indices[ndims - 1] += 1;
    for (int k = ndims - 1; k > 0; k--) {
      if (indices[k] >= shape[k]) {
        indices[k] = 0;
        indices[k - 1] += 1;
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

  Tensor A = make_tensor("A", {2, 2}, true, {1, 2, 3, 4}, ctx);
  Tensor B = expand(A, {2, 2, 2});
  Tensor C = make_tensor("C", {2, 2, 2}, true, {1, 2, 3, 4, 5, 6, 7, 8}, ctx);
  Tensor D = B + C;
  engine.backward(D);
  printTensor(D->hostGradBuffer().data(), D->getShape(), D->getStrides());
  printTensor(C->hostGradBuffer().data(), C->getShape(), C->getStrides());
  printTensor(B->hostGradBuffer().data(), B->getShape(), B->getStrides());
  printTensor(A->hostGradBuffer().data(), A->getShape(), A->getStrides());
  return 0;
}
