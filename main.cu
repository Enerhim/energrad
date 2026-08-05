#include <autograd.cuh>
#include <tensor.cuh>
#include <utils.cuh>

int main() {

  Tensor t1 = TensorRand({2, 2, 2}, StorageDevice::CPU, true, -100.0f, 100.0f);

  print(t1);
  return 0;
}
