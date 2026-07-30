#include <tensor.cuh>
// #include "../../../include/utils.cuh"

// NOTE: CUDA Kernels

// NOTE: CPU Kernels

// NOTE: API Functions

Tensor TensorBroadcast(Tensor A, std::array<size_t, MAX_DIMS> targetShape) {
  auto storageA = A->storage;
  size_t rankA = A->rank;
  size_t rankTarget = targetShape.size();

  auto result = TensorInit(targetShape, rankA, 0.0f, storageA->device, false);
  result->storage = storageA;

  std::array<size_t, MAX_DIMS> &stridesA = A->strides;
  std::array<size_t, MAX_DIMS> &stridesResult = result->strides;

  size_t zeroPadding = rankTarget - rankA;
  std::fill(stridesResult.begin(), stridesResult.begin() + zeroPadding, 0);
  std::copy(stridesA.begin(), stridesA.begin() + rankA,
            stridesResult.begin() + zeroPadding);

  for (size_t i = zeroPadding; i < rankTarget; i++) {
    auto shapeI = A->shape[i - zeroPadding];
    if (shapeI != targetShape[i] && shapeI == 1)
      stridesResult[i] = 0;
  }

  return result;
}

Tensor TensorTranspose(Tensor A, size_t dim1, size_t dim2) {
  size_t rankA = A->rank;
  if (dim1 >= rankA || dim2 >= rankA)
    return nullptr;

  if (dim1 == dim2)
    return TensorCopy(A, false);

  auto shape = A->shape;
  auto strides = A->strides;

  std::swap(shape[dim1], shape[dim2]);
  std::swap(strides[dim1], strides[dim2]);

  auto result = TensorInit(shape, rankA, 0.0f, A->storage->device, false);
  result->strides = strides;
  result->storage = A->storage;

  return result;
}

// All op outputs are already contiguous
Tensor TensorContiguous(Tensor A) { return TensorScale(A, 1.0f); }
