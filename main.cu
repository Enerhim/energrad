#include <autograd.cuh>
#include <tensor.cuh>
#include <utils.cuh>

int main() {

  Tensor t1 = TensorRand({2, 2, 2}, StorageDevice::CPU, false, -100.0f, 100.0f);
  Tensor t2 = TensorRand({2, 2, 2}, StorageDevice::CPU, false, -40.0f, 40.0f);
  Node a = CreateNode(t1, true, true);
  Node b = CreateNode(t2, true, true);

  std::cout << "a: ";
  print(a->data);
  std::cout << "b: ";
  print(b->data);
  Node result = MulNode(a, b);

  std::cout << "Data after op: ";
  print(result->data);

  result->grad = TensorOnes({2, 2, 2}, StorageDevice::CPU);
  std::cout << "Allocated grad: ";
  print(result->grad);

  MulBackward(result);

  std::cout << "Grad parent a after backward pass: ";
  print(a->grad);
  std::cout << "Grad parent b after backward pass: ";
  print(b->grad);
  return 0;
}
