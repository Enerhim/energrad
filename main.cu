#include <autograd.cuh>
#include <tensor.cuh>
#include <utils.cuh>

int main() {
  Tensor t1 = TensorCreate({2, 2, 2}, 1.5f, StorageDevice::CPU);
  Tensor t2 = TensorCreate({2, 2, 2}, -1.6f, StorageDevice::CPU);

  Node a = CreateNode(t1, true, true);
  Node b = CreateNode(t2, true, true);

  Node result = MulNode(a, b);

  std::cout << "Data after addition: ";
  print(result->data);

  result->grad = TensorOnes({2, 2, 2}, StorageDevice::CPU);
  std::cout << "Allocated grad: ";

  print(result->grad);
  MulBackward(result);

  std::cout << "Grad A after backward pass: ";
  print(a->grad);
  std::cout << "Grad B after backward pass: ";
  print(b->grad);

  return 0;
}
