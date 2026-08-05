#include <autograd.cuh>
#include <tensor.cuh>
#include <utils.cuh>

int main() {

  Tensor t1 = TensorCreate({2, 2, 2, 2}, 1.5f, StorageDevice::CPU);
  Tensor t2 = TensorCreate({2, 2, 2, 2}, 2.0f, StorageDevice::CPU);
  Node a = CreateNode(t1, true, true);
  Node b = CreateNode(t2, true, true);

  Node result = MatmulNode(a, b);

  std::cout << "Data after op: ";
  print(result->data);

  result->grad = TensorOnes({2, 2, 2, 2}, StorageDevice::CPU);
  std::cout << "Allocated grad: ";
  print(result->grad);

  MatmulBackward(result);

  std::cout << "Grad parent after backward pass: ";
  print(a->grad);
  std::cout << "Grad parent after backward pass: ";
  print(b->grad);
  return 0;
}
