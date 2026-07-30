#include <tensor.cuh>
#include <utils.cuh>

int main() {
  auto t1 =
      TensorCreate(std::vector<size_t>{1, 2, 3}, 1.0f, StorageDevice::CPU);
  print(t1);
  return 0;
}
