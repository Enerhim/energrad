#include <iostream>
#include <pcg/pcg_random.hpp>
#include <random>
#include <tensor.cuh>
#include <utils.cuh>

// NOTE: CUDA Kernels

// NOTE: CPU Kernels

void CPU_TensorFill(float *dataA, float a, TensorMeta &metaA) {
  for (size_t idx = 0; idx < metaA.elements; idx++) {
    dataA[idx] = a;
  }
}

// NOTE: API Functions

// TODO: Tensor Serialization

Tensor TensorInit(std::array<size_t, MAX_DIMS> &shape, size_t rank, float a,
                  StorageDevice device, bool makeStorage) {
  auto result = std::make_shared<TensorView>(shape, rank);

  if (makeStorage) {
    size_t elements = calculateLogicalElementNo(shape, rank);
    auto storageResult =
        std::make_shared<TensorStorage>(elements * sizeof(float), device);
    result->storage = storageResult;

    TensorMeta meta(static_cast<int>(rank), elements, result->shape,
                    result->strides);

    switch (device) {
    case StorageDevice::CPU:
      CPU_TensorFill(storageResult->data, a, meta);
      break;
    case StorageDevice::CUDA:
      // TODO: Cuda tensor init
      break;
    }
  }

  return result;
}

Tensor TensorCreate(const std::vector<size_t> &s, float a,
                    StorageDevice device) {
  size_t rank = s.size();
  std::array<size_t, MAX_DIMS> shapeArray;
  std::copy(s.begin(), s.end(), shapeArray.begin());
  std::fill(shapeArray.begin() + rank, shapeArray.end(), 0);

  return TensorInit(shapeArray, rank, a, device, true);
}

Tensor TensorCopy(Tensor A, bool newStorage) {
  size_t rankA = A->rank;
  auto storageA = A->storage;
  auto shapeA = A->shape, stridesA = A->strides;
  auto result = TensorInit(shapeA, rankA, 0.0f, storageA->device, newStorage);

  if (newStorage) {
    TensorMeta meta(rankA, calculateLogicalElementNo(shapeA, rankA), shapeA,
                    stridesA);
    CPU_TensorScale(storageA->data, meta, 1.0f, result->storage->data);
  } else {
    result->storage = storageA;
  }

  return result;
}

void TensorLoadData(Tensor A, const float *data, size_t size_) {
  if (size_ != (A->storage->bytes / sizeof(float)))
    return;

  switch (A->storage->device) {
  case StorageDevice::CPU:
    std::copy(data, data + size_, A->storage->data);
    break;
  case StorageDevice::CUDA:
    // Load data to GPU
    break;
  }
}

Tensor TensorOnes(const std::vector<size_t> &shape, StorageDevice device) {
  return TensorCreate(shape, 1.0f, device);
}

Tensor TensorZeros(const std::vector<size_t> &shape,
                   const StorageDevice device) {
  return TensorCreate(shape, 0.0f, device);
}

Tensor TensorRand(const std::vector<size_t> &shape, const StorageDevice device,
                  bool normal, float min, float max) {
  Tensor t = TensorCreate(shape, 0.0f, device);
  size_t elems = calculateLogicalElementNo(t->shape, t->rank);
  pcg32 rng;
  if (!normal) {
    std::uniform_real_distribution<float> uniform(min, max);
    std::vector<float> nums(elems, 0.0f);
    for (float &x : nums)
      x = uniform(rng);
    TensorLoadData(t, nums.data(), elems);
  } else {
    std::normal_distribution<float> normal(min, max);
    std::vector<float> nums(elems, 0.0f);
    for (float &x : nums)
      x = normal(rng);
    TensorLoadData(t, nums.data(), elems);
  }

  return t;
}

// TODO: Tensor Move Device (Will do with CUDA Kernels)

void TensorMoveDevice(Tensor A, StorageDevice device) {}
