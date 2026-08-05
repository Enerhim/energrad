#include <tensor.cuh>

TensorView::TensorView(std::array<size_t, MAX_DIMS> s, size_t rank)
    : rank(rank), shape(s) {
  size_t current_stride = 1;
  for (int dim = static_cast<int>(rank) - 1; dim >= 0; dim--) {
    strides[dim] = current_stride;
    current_stride *= shape[dim];
  }
  // NOTE: Do not auto allocate, make a flag that allocate only when
  // specifically told to
};

TensorStorage::TensorStorage(size_t bytes, StorageDevice device)
    : bytes(bytes), device(device) {
  switch (device) {
  case StorageDevice::CPU:
    data = new float[bytes / sizeof(float)];
    break;
  case StorageDevice::CUDA:
    break;
  }
};

TensorStorage::~TensorStorage() {
  switch (device) {
  case StorageDevice::CPU:
    delete[] data;
    break;

  case StorageDevice::CUDA:
    // TODO: Impl CUDA Free
    break;
  }
}

TensorMeta::TensorMeta() {};
