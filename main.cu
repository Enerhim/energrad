#include <autograd.cuh>
#include <tensor.cuh>
#include <utils.cuh>

int main() {
  Tensor t = TensorCreate({2, 2, 2}, 2.0f, StorageDevice::CPU);
  auto r = TensorSum(t, 1, false);
  print(t);
  print(r);

  return 0;
}
