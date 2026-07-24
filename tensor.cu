#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

class TensorStorage;
class TensorView;

using Tensor = std::shared_ptr<TensorView>;
using StoragePtr = std::shared_ptr<TensorStorage>;

#define MAX_DIMS 10
enum class StorageDevice { CPU, CUDA };

class TensorView : public std::enable_shared_from_this<TensorView> {
public:
  TensorView(std::span<const size_t> s) {
    rank = s.size();
    std::copy(s.begin(), s.end(), shape.begin());
    std::fill(shape.begin() + rank, shape.end(), 0);

    size_t current_stride = 1;
    for (int dim = static_cast<int>(rank) - 1; dim >= 0; dim--) {
      strides[dim] = current_stride;
      current_stride *= shape[dim];
    }
    // NOTE: Do not auto allocate, make a flag that allocate only when
    // specifically told to
  };
  size_t rank;
  std::array<size_t, MAX_DIMS> shape;
  std::array<size_t, MAX_DIMS> strides;
  std::shared_ptr<TensorStorage> storage;
};

class TensorStorage {
public:
  TensorStorage(size_t bytes, StorageDevice device)
      : bytes(bytes), device(device) {
    switch (device) {
    case StorageDevice::CPU:
      data = new float[bytes / sizeof(float)];
      break;
    case StorageDevice::CUDA:
      // TODO: Impl CUDA Allocation
      break;
    }
  };
  ~TensorStorage() {
    switch (device) {
    case StorageDevice::CPU:
      delete[] data;
      break;

    case StorageDevice::CUDA:
      // TODO: Impl CUDA Free
      break;
    }
  }
  size_t bytes;
  float *data = nullptr;

  StorageDevice device;
};

typedef struct TensorMeta {

  TensorMeta(int ndims, size_t elements, std::array<size_t, MAX_DIMS> shape,
             std::array<size_t, MAX_DIMS> strides)
      : ndims(ndims), elements(elements), shape(shape), strides(strides) {};

  size_t ndims;
  size_t elements;
  std::array<size_t, MAX_DIMS> shape;
  std::array<size_t, MAX_DIMS> strides;
} TensorMeta;

Tensor TensorLoad(std::string filename);
void TensorSave(Tensor A, std::string filename);

Tensor TensorInit(std::vector<size_t> &shape, float a, StorageDevice device);
Tensor TensorCopy(Tensor A, bool newStorage = false);
void TensorMoveDevice(Tensor A, StorageDevice device);

// These ops read in a strided way and make contiguous tensors

Tensor TensorScale(Tensor A, float a);
Tensor TensorRelu(Tensor A);
Tensor TensorTanh(Tensor A);

Tensor TensorAdd(Tensor A, Tensor B);
Tensor TensorSub(Tensor A, Tensor B);

Tensor TensorMatmul(Tensor A, Tensor B, bool AT = false, bool BT = false);
Tensor TensorSoftmax(Tensor A);

// Stride wreckers - will make the tensor non-contiguous

Tensor TensorTranspose(Tensor A, size_t dim1, size_t dim2);
Tensor TensorBroadcast(Tensor A, std::span<size_t> targetShape);
Tensor TensorContiguous(Tensor A);
void CPU_TensorFill(float *dataA, TensorMeta &metaA);
void CPU_TensorScale(float *dataA, TensorMeta &metaA, float scalar,
                     float *dataResult);

void CPU_TensorRelu(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorTanh(float *dataA, TensorMeta &metaA, float *dataResult);

void CPU_TensorAdd(float *dataA, TensorMeta &metaA, float *dataB,
                   TensorMeta &metaB, float *dataResult);
void CPU_TensorSub(float *dataA, TensorMeta &metaA, float *dataB,
                   TensorMeta &metaB, float *dataResult);

void CPU_TensorMatmulRR(float *dataA, TensorMeta &metaA, float *dataB,
                        TensorMeta &metaB, float *dataResult);
void CPU_TensorMatmulRT(float *dataA, TensorMeta &metaA, float *dataB,
                        TensorMeta &metaB, float *dataResult);
void CPU_TensorMatmulTR(float *dataA, TensorMeta &metaA, float *dataB,
                        TensorMeta &metaB, float *dataResult);
void CPU_TensorMatmulTT(float *dataA, TensorMeta &metaA, float *dataB,
                        TensorMeta &metaB, float *dataResult);

void CPU_TensorSoftmax(float *dataA, TensorMeta &metaA, float *dataResult);

// Utils
bool checkBroadcastable(Tensor A, Tensor target) {
  auto shapeA = A->shape, shapeTarget = target->shape;
  auto rankA = A->rank, rankTarget = target->rank;

  if (rankA > rankTarget)
    return false;

  int i = static_cast<int>(rankTarget) - 1, j = static_cast<int>(rankA) - 1;
  while (j >= 0) {
    if (shapeA[j] != shapeTarget[i] && shapeA[j] != 1)
      return false;

    --i, --j;
  }

  return true;
}

size_t calculateLogicalElementNo(std::array<size_t, MAX_DIMS> &shape,
                                 size_t rank) {
  size_t elements = 1;
  for (int i = 0; i < rank; i++) {
    elements *= shape[i];
  }
  return elements;
}

bool checkContiguous(Tensor A) {
  size_t rank = A->rank;
  if (rank == 0)
    return true;

  size_t currentStride = 1;
  for (int dim = static_cast<int>(rank) - 1; dim >= 0; dim--) {
    if (A->strides[dim] != currentStride)
      return false;
    currentStride *= A->shape[dim];
  }

  return true;
}

void print(Tensor A) {
  auto shape = A->shape;
  auto strides = A->strides;
  auto rank = A->rank;
  auto data = A->storage->data;
  auto elements = calculateLogicalElementNo(shape, rank);

  std::cout << "[ ";
  for (int idx = 0; idx < elements; idx++) {
    size_t offset = 0;
    size_t remaining = idx;

    for (int dim = static_cast<int>(rank) - 1; dim >= 0; dim--) {
      size_t indice = remaining % shape[dim];
      remaining /= shape[dim];

      offset += indice * strides[dim];
    }
    std::cout << data[offset];
  }
  std::cout << " ]" << std::endl;
}

// Tensor Ops

Tensor TensorInit(std::vector<size_t> &shape, float a, StorageDevice device) {
  size_t elements = 1;
  size_t rank = shape.size();
  for (int i = 0; i < rank; i++) {
    elements *= shape[i];
  }
  auto storageResult = std::make_shared<TensorStorage>(elements * sizeof(float),
                                                       StorageDevice::CPU);
  auto result = std::make_shared<TensorView>(shape);
  TensorMeta meta(static_cast<int>(rank), elements, result->shape,
                  result->strides);

  switch (device) {
  case StorageDevice::CPU:
    CPU_TensorFill(storageResult->data, meta);
    break;
  case StorageDevice::CUDA:
    // TODO: Cuda tensor init
    break;
  }

  return result;
}
Tensor TensorCopy(Tensor A, bool newStorage) {
  auto result = std::make_shared<TensorView>(A->shape);
  auto storageA = A->storage;
  auto shapeA = A->shape, stridesA = A->strides;
  size_t rankA = A->rank;
  if (newStorage) {
    result->storage =
        std::make_shared<TensorStorage>(storageA->bytes, storageA->device);

    size_t elements = calculateLogicalElementNo(shapeA, rankA);
    TensorMeta meta(rankA, elements, shapeA, stridesA);
    CPU_TensorScale(storageA->data, meta, 1.0f, result->storage->data);
  } else {
    result->storage = storageA;
  }

  return result;
}
Tensor TensorScale(Tensor A, float a) {
  auto storageA = A->storage;
  auto dataA = storageA->data;
  auto rankA = A->rank;
  auto shapeA = A->shape, stridesA = A->strides;

  // Logical number of elements of A
  size_t elementsA = calculateLogicalElementNo(shapeA, rankA);
  TensorMeta meta(rankA, elementsA, shapeA, stridesA);

  auto storageResult = std::make_shared<TensorStorage>(
      elementsA * sizeof(float), storageA->device);
  auto result =
      std::make_shared<TensorView>(std::span<size_t>(shapeA.data(), rankA));
  result->storage = storageResult;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorScale(dataA, meta, a, storageResult->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorRelu(Tensor A) {
  auto storageA = A->storage;
  auto dataA = storageA->data;
  auto rankA = A->rank;
  auto shapeA = A->shape, stridesA = A->strides;

  // Logical number of elements of A
  size_t elementsA = calculateLogicalElementNo(shapeA, rankA);
  TensorMeta meta(rankA, elementsA, shapeA, stridesA);

  auto storageResult = std::make_shared<TensorStorage>(
      elementsA * sizeof(float), storageA->device);
  auto result =
      std::make_shared<TensorView>(std::span<size_t>(shapeA.data(), rankA));
  result->storage = storageResult;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorRelu(dataA, meta, storageResult->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorTanh(Tensor A) {
  auto storageA = A->storage;
  auto dataA = storageA->data;
  auto rankA = A->rank;
  auto shapeA = A->shape, stridesA = A->strides;

  // Logical number of elements of A
  size_t elementsA = calculateLogicalElementNo(shapeA, rankA);
  TensorMeta meta(rankA, elementsA, shapeA, stridesA);

  auto storageResult = std::make_shared<TensorStorage>(
      elementsA * sizeof(float), storageA->device);
  auto result =
      std::make_shared<TensorView>(std::span<size_t>(shapeA.data(), rankA));
  result->storage = storageResult;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorTanh(dataA, meta, storageResult->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorAdd(Tensor A, Tensor B) {
  auto storageA = A->storage, storageB = B->storage;
  auto dataA = storageA->data, dataB = storageB->data;

  // If different devices, move to the device of the first matrix
  auto deviceA = storageA->device;
  if (deviceA != storageB->device) {
    TensorMoveDevice(B, deviceA);
    storageB = A->storage;
    dataB = storageB->data;
  }

  // Create Meta
  auto shapeA = A->shape, stridesA = A->strides, shapeB = B->shape,
       stridesB = B->strides;
  auto rankA = A->rank, rankB = B->rank;
  size_t elementsA = calculateLogicalElementNo(shapeA, rankA),
         elementsB = calculateLogicalElementNo(shapeB, B->rank);

  if (shapeA != shapeB) {
    if (checkBroadcastable(A, B)) {
      A = TensorBroadcast(A, B->shape);
    }
  }

  TensorMeta metaA(rankA, elementsA, shapeA, stridesA);
  TensorMeta metaB(rankB, elementsB, shapeB, stridesB);

  // Make result
  auto storageResult =
      std::make_shared<TensorStorage>(elementsA * sizeof(float), deviceA);
  auto result =
      std::make_shared<TensorView>(std::span<size_t>(shapeA.data(), rankA));
  result->storage = storageResult;

  switch (deviceA) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorAdd(dataA, metaA, dataB, metaB, storageResult->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorSub(Tensor A, Tensor B) {
  auto storageA = A->storage, storageB = B->storage;
  auto dataA = storageA->data, dataB = storageB->data;
  auto rankA = A->rank, rankB = B->rank;
  // If different devices, move to the device of the first matrix
  auto deviceA = storageA->device;
  if (deviceA != storageB->device) {
    TensorMoveDevice(B, deviceA);
    storageB = B->storage;
    dataB = storageB->data;
  }

  // Create Meta
  auto shapeA = A->shape, stridesA = A->strides, shapeB = B->shape,
       stridesB = B->strides;

  size_t elementsA = calculateLogicalElementNo(shapeA, rankA),
         elementsB = calculateLogicalElementNo(shapeB, B->rank);

  if (shapeA != shapeB) {
    if (checkBroadcastable(A, B)) {
      A = TensorBroadcast(A, B->shape);
    }
  }

  // Make result
  auto storageResult =
      std::make_shared<TensorStorage>(elementsA * sizeof(float), deviceA);
  auto result =
      std::make_shared<TensorView>(std::span<size_t>(shapeA.data(), rankA));
  result->storage = storageResult;
  // TODO: Broadcast if in scope

  TensorMeta metaA(rankA, elementsA, shapeA, stridesA);
  TensorMeta metaB(rankB, elementsB, shapeB, stridesB);

  switch (deviceA) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorSub(dataA, metaA, dataB, metaB, storageResult->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

// TODO: Batch matmul
Tensor TensorMatmul(Tensor A, Tensor B, bool AT, bool BT) {
  uint8_t transposeFlags = (AT << 1) + BT;
  auto rankA = A->rank, rankB = B->rank;
  auto shapeA = A->shape, stridesA = A->strides, shapeB = B->shape,
       stridesB = B->strides;

  if (rankA < 2 || rankB < 2)
    return nullptr;

  auto targetShape = std::span(shapeA.data(), rankA);
  targetShape[rankA - 1] = shapeB[rankB - 1];
  targetShape[rankA - 2] = shapeB[rankB - 2];
  B = TensorContiguous(TensorBroadcast(B, targetShape));

  if (!checkContiguous(A)) {
    A = TensorContiguous(A);
  }
  if (!checkContiguous(B)) {
    B = TensorContiguous(B);
  }

  auto storageA = A->storage, storageB = B->storage;
  auto dataA = storageA->data, dataB = storageB->data;

  // If different devices, move to the device of the first matrix
  auto deviceA = storageA->device;
  if (deviceA != storageB->device) {
    TensorMoveDevice(B, deviceA);
    storageB = B->storage;
    dataB = storageB->data;
  }

  // Create Meta
  size_t elementsA = calculateLogicalElementNo(shapeA, rankA),
         elementsB = calculateLogicalElementNo(shapeB, rankB);

  // Make result
  auto storageResult =
      std::make_shared<TensorStorage>(elementsA * sizeof(float), deviceA);
  auto result =
      std::make_shared<TensorView>(std::span<size_t>(shapeA.data(), rankA));
  result->storage = storageResult;
  // TODO: Broadcast if in scope

  TensorMeta metaA(rankA, elementsA, shapeA, stridesA);
  TensorMeta metaB(rankB, elementsB, shapeB, stridesB);

  switch (deviceA) {
  case StorageDevice::CPU:
    CPU_TensorMatmulRR(dataA, metaA, dataB, metaB, storageResult->data);
    // switch (transposeFlags) {
    // case 0b00:
    //   CPU_TensorMatmulRR(dataA, metaA, dataB, metaB, storageResult->data);
    //   break;
    // case 0b01:
    //   CPU_TensorMatmulRT(dataA, metaA, dataB, metaB, storageResult->data);
    //   break;
    // case 0b10:
    //   CPU_TensorMatmulTR(dataA, metaA, dataB, metaB, storageResult->data);
    //   break;
    // case 0b11:
    //   CPU_TensorMatmulTT(dataA, metaA, dataB, metaB, storageResult->data);
    //   break;
    // }
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorSoftmax(Tensor A) {
  auto storageA = A->storage;
  auto dataA = storageA->data;
  auto rankA = A->rank;
  auto shapeA = A->shape, stridesA = A->strides;

  // Logical number of elements of A
  size_t elementsA = calculateLogicalElementNo(shapeA, rankA);
  TensorMeta meta(rankA, elementsA, shapeA, stridesA);

  auto storageResult = std::make_shared<TensorStorage>(
      elementsA * sizeof(float), storageA->device);
  auto result =
      std::make_shared<TensorView>(std::span<size_t>(shapeA.data(), rankA));
  result->storage = storageResult;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorSoftmax(dataA, meta, storageResult->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

// These ones f up the strides

Tensor TensorBroadcast(Tensor A, std::span<size_t> targetShape) {
  auto storageA = A->storage;
  size_t rankA = A->rank;
  size_t rankTarget = targetShape.size();

  auto result = std::make_shared<TensorView>(targetShape);
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
    // Wrong af by your architecture. Make a TensorCopy function for that shit
    return TensorCopy(A, false);

  auto shape = A->shape;
  auto strides = A->strides;

  std::swap(shape[dim1], shape[dim2]);
  std::swap(strides[dim1], strides[dim2]);

  auto result = std::make_shared<TensorView>(shape);
  result->strides = strides;
  result->storage = A->storage;

  return result;
}

Tensor TensorContiguous(Tensor A) {
  auto storageA = A->storage;
  auto dataA = storageA->data;
  auto rankA = A->rank;
  auto shapeA = A->shape, stridesA = A->strides;

  // Logical number of elements of A
  size_t elementsA = calculateLogicalElementNo(shapeA, rankA);
  TensorMeta meta(rankA, elementsA, shapeA, stridesA);

  auto storageResult = std::make_shared<TensorStorage>(
      elementsA * sizeof(float), storageA->device);
  auto result =
      std::make_shared<TensorView>(std::span<size_t>(shapeA.data(), rankA));
  result->storage = storageResult;

  switch (storageA->device) {
  case StorageDevice::CPU:
    CPU_TensorScale(dataA, meta, 1.0f, storageResult->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

// CPU Operations
// Kernels will only work with the data, parameters and required
// context. They will only perform their function - no broadcasting or different
// shape BS

void CPU_TensorScale(float *dataA, TensorMeta &metaA, float scalar,
                     float *dataResult) {
  // Need to unfold according to strides
  size_t ndim = metaA.ndims;

  for (size_t idx = 0; idx < metaA.elements; idx++) {
    size_t offset = 0;
    size_t remaining = idx;

    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
      size_t dim_size = metaA.shape[dim];
      size_t indice = remaining % dim_size;
      remaining /= dim_size;

      offset += indice * metaA.strides[dim];
    }

    dataResult[idx] = dataA[offset] * scalar;
  }
}

void CPU_TensorRelu(float *dataA, TensorMeta &metaA, float *dataResult) {
  // Need to unfold according to strides
  size_t ndim = metaA.ndims;

  for (size_t idx = 0; idx < metaA.elements; idx++) {
    size_t offset = 0;
    size_t remaining = idx;

    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
      size_t dim_size = metaA.shape[dim];
      size_t indice = remaining % dim_size;
      remaining /= dim_size;

      offset += indice * metaA.strides[dim];
    }

    dataResult[idx] = std::max(dataA[offset], 0.0f);
  }
}

void CPU_TensorTanh(float *dataA, TensorMeta &metaA, float *dataResult) {
  // Need to unfold according to strides
  size_t ndim = metaA.ndims;

  for (size_t idx = 0; idx < metaA.elements; idx++) {
    size_t offset = 0;
    size_t remaining = idx;

    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
      size_t dim_size = metaA.shape[dim];
      size_t indice = remaining % dim_size;
      remaining /= dim_size;

      offset += indice * metaA.strides[dim];
    }

    dataResult[idx] = std::tanhf(dataA[offset]);
  }
}

// Shape must be same here
void CPU_TensorAdd(float *dataA, TensorMeta &metaA, float *dataB,
                   TensorMeta &metaB, float *dataResult) {

  size_t ndim = metaA.ndims;
  for (size_t idx = 0; idx < metaA.elements; idx++) {
    size_t offsetA = 0;
    size_t offsetB = 0;

    size_t remaining = idx;

    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
      size_t dim_size = metaA.shape[dim];
      size_t indice = remaining % dim_size;
      remaining /= dim_size;

      offsetA += indice * metaA.strides[dim];
      offsetB += indice * metaB.strides[dim];
    }

    dataResult[idx] = dataA[offsetA] + dataB[offsetB];
  }
}

void CPU_TensorSub(float *dataA, TensorMeta &metaA, float *dataB,
                   TensorMeta &metaB, float *dataResult) {

  size_t ndim = metaA.ndims;
  for (size_t idx = 0; idx < metaA.elements; idx++) {
    size_t offsetA = 0;
    size_t offsetB = 0;

    size_t remaining = idx;

    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
      size_t dim_size = metaA.shape[dim];
      size_t indice = remaining % dim_size;
      remaining /= dim_size;

      offsetA += indice * metaA.strides[dim];
      offsetB += indice * metaB.strides[dim];
    }

    dataResult[idx] = dataA[offsetA] - dataB[offsetB];
  }
}

void CPU_TensorMatmulRR(float *dataA, TensorMeta &metaA, float *dataB,
                        TensorMeta &metaB, float *dataResult) {
  assert(metaA.shape[metaA.ndims - 1] == metaB.shape[metaB.ndims - 2]);

  size_t M = metaA.shape[metaA.ndims - 2];
  size_t K = metaA.shape[metaA.ndims - 1];
  size_t N = metaB.shape[metaB.ndims - 1];

  size_t matrixSizeA = M * K;
  size_t matrixSizeB = K * N;
  size_t matrixSizeC = M * N;

  size_t batchCount = metaA.elements / matrixSizeA;

  for (size_t b = 0; b < batchCount; b++) {
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        float sum_ = 0.0f;
        for (size_t k = 0; k < K; k++) {
          sum_ += dataA[b * matrixSizeA + i * K + k] *
                  dataB[b * matrixSizeB + k * N + j];
        }
        dataResult[b * matrixSizeC + i * N + j] = sum_;
      }
    }
  }
}

void CPU_TensorSoftmax(float *dataA, TensorMeta &metaA, float *dataResult) {
  size_t M = metaA.shape[metaA.ndims - 2];
  size_t N = metaA.shape[metaA.ndims - 1];

  size_t matrixSizeA = M * N;
  size_t batchCount = metaA.elements / matrixSizeA;

  for (size_t b = 0; b < batchCount; b++) {
    size_t offset = matrixSizeA * b;
    for (size_t i = 0; i < M; i++) {
      // Calculate max to reduce numerical instability
      float max_val = dataA[offset + i * N];
      for (size_t j = 0; j < N; j++)
        max_val = std::max(dataA[offset + i * N + j], max_val);

      float sum_ = 0.0f;
      for (size_t j = 0; j < N; j++)
        sum_ += std::expf(dataA[offset + i * N + j] - max_val);

      for (size_t j = 0; j < N; j++)
        dataResult[offset + i * N + j] =
            std::expf(dataA[offset + i * N + j] - max_val) / sum_;
    }
  }
}
