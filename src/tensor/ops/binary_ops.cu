#include <tensor.cuh>
#include <utils.cuh>

// NOTE: CUDA Kernels

// NOTE: CPU Kernels

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

void CPU_TensorMul(float *dataA, TensorMeta &metaA, float *dataB,
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
    dataResult[idx] = dataA[offsetA] * dataB[offsetB];
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

// NOTE: API Functions

void _TensorBinary_(Tensor A, Tensor B, Tensor &result, TensorMeta &mA,
                    TensorMeta &mB, bool autoBroadcast, bool autoBroadcastBoth,
                    bool autoMove) {
  auto storageA = A->storage, storageB = B->storage;
  auto dataA = storageA->data, dataB = storageB->data;
  auto rankA = A->rank, rankB = B->rank;
  auto shapeA = A->shape, stridesA = A->strides, shapeB = B->shape,
       stridesB = B->strides;

  auto deviceA = storageA->device;
  if (autoMove) {
    if (deviceA != storageB->device) {
      TensorMoveDevice(B, deviceA);
      storageB = B->storage;
      dataB = storageB->data;
    }
  }

  if (autoBroadcast) {
    if (shapeA != shapeB) {
      if (checkBroadcastable(B, A)) {
        B = TensorBroadcast(B, shapeA);
        shapeB = B->shape, stridesB = B->strides;

      } else if (autoBroadcastBoth) {
        if (checkBroadcastable(A, B)) {
          A = TensorBroadcast(A, shapeB);
          shapeA = A->shape, stridesA = A->strides;
        }
      }
    }
  }

  size_t elementsA = calculateLogicalElementNo(shapeA, rankA),
         elementsB = calculateLogicalElementNo(shapeB, rankB);

  mA = TensorMeta(rankA, elementsA, shapeA, stridesA);
  mB = TensorMeta(rankB, elementsB, shapeB, stridesB);
  result = TensorInit(shapeA, rankA, 0.0f, deviceA, true);
}

Tensor TensorAdd(Tensor A, Tensor B) {
  Tensor result;
  TensorMeta metaA, metaB;
  _TensorBinary_(A, B, result, metaA, metaB, true, true, true);
  auto storageA = A->storage;
  auto deviceA = storageA->device;
  auto dataA = storageA->data, dataB = B->storage->data;
  switch (deviceA) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorAdd(dataA, metaA, dataB, metaB, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorSub(Tensor A, Tensor B) {
  Tensor result;
  TensorMeta metaA, metaB;
  _TensorBinary_(A, B, result, metaA, metaB, true, true, true);
  auto storageA = A->storage;
  auto deviceA = storageA->device;
  auto dataA = storageA->data, dataB = B->storage->data;
  switch (deviceA) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorSub(dataA, metaA, dataB, metaB, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }
  return result;
}

Tensor TensorMul(Tensor A, Tensor B) {
  Tensor result;
  TensorMeta metaA, metaB;
  _TensorBinary_(A, B, result, metaA, metaB, true, true, true);
  auto storageA = A->storage;
  auto deviceA = storageA->device;
  auto dataA = storageA->data, dataB = B->storage->data;
  switch (deviceA) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorMul(dataA, metaA, dataB, metaB, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }
  return result;
}
Tensor TensorMatmul(Tensor A, Tensor B, bool AT, bool BT) {
  // uint8_t transposeFlags = (AT << 1) + BT;
  Tensor result;
  TensorMeta metaA, metaB;
  _TensorBinary_(A, B, result, metaA, metaB, false, false, true);

  auto rankA = A->rank, rankB = B->rank;

  if (rankA < 2 || rankB < 2)
    return nullptr;

  auto targetShape = metaA.shape;
  targetShape[rankA - 1] = metaB.shape[rankB - 1];
  targetShape[rankA - 2] = metaB.shape[rankB - 2];
  B = TensorContiguous(TensorBroadcast(B, targetShape));

  if (!checkContiguous(A)) {
    A = TensorContiguous(A);
  }
  if (!checkContiguous(B)) {
    B = TensorContiguous(B);
  }

  auto storageA = A->storage, storageB = B->storage;
  auto dataA = storageA->data, dataB = storageB->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    CPU_TensorMatmulRR(dataA, metaA, dataB, metaB, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}
