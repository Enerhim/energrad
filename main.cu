#include <tensor.cuh>
#include <utils.cuh>

int main() {
  auto t1 =
      TensorCreate(std::vector<size_t>{1, 2, 3}, 1.5f, StorageDevice::CPU);
  auto t2 =
      TensorCreate(std::vector<size_t>{1, 2, 3}, 1.0f, StorageDevice::CPU);
  auto t3 = TensorZeros(std::vector<size_t>{4, 1, 2, 2}, StorageDevice::CPU);

  auto t4 = TensorOnes(std::vector<size_t>{4, 1, 2, 2}, StorageDevice::CPU);

  print(t1);
  print(t2);
  print(t3);

  print(TensorNeg(t1));
  print(TensorAbs(t1));
  print(TensorSqrt(t1));
  print(TensorReciprocal(t1));
  print(TensorExp(t1));
  print(TensorNLog(t1));
  print(TensorLog2(t1));
  print(TensorLog10(t1));
  print(TensorSin(t1));
  print(TensorSinh(t1));
  print(TensorCos(t1));
  print(TensorCosh(t1));
  print(TensorTan(t1));
  print(TensorTanh(t1));
  print(TensorFloor(t1));
  print(TensorCeil(t1));
  print(TensorScale(t1, 3.0f));
  print(TensorRelu(t1));
  print(TensorSigmoid(t1));
  print(TensorSoftmax(t1));

  return 0;
}
