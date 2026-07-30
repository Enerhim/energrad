#include <tensor.cuh>
#include <utils.cuh>

// NOTE: CUDA Kernel

// NOTE: CPU Kernel

void CPU_TensorNeg(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = -dataA[offset];
  }
}

void CPU_TensorAbs(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::abs(dataA[offset]);
  }
}

void CPU_TensorSqrt(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::sqrt(dataA[offset]);
  }
}

void CPU_TensorReciprocal(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = 1.0f / dataA[offset];
  }
}

void CPU_TensorExp(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::expf(dataA[offset]);
  }
}

void CPU_TensorNLog(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::logf(dataA[offset]);
  }
}

void CPU_TensorLog2(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::log2f(dataA[offset]);
  }
}

void CPU_TensorLog10(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::log10f(dataA[offset]);
  }
}

void CPU_TensorSin(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::sinf(dataA[offset]);
  }
}

void CPU_TensorCos(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::cosf(dataA[offset]);
  }
}

void CPU_TensorSinh(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::sinhf(dataA[offset]);
  }
}

void CPU_TensorCosh(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::coshf(dataA[offset]);
  }
}

void CPU_TensorTan(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::tanf(dataA[offset]);
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

void CPU_TensorFloor(float *dataA, TensorMeta &metaA, float *dataResult) {
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
    dataResult[idx] = std::floorf(dataA[offset]);
  }
}

void CPU_TensorCeil(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    dataResult[idx] = std::ceilf(dataA[offset]);
  }
}

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

void CPU_TensorSigmoid(float *dataA, TensorMeta &metaA, float *dataResult) {
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

    float x = dataA[offset];
    float sigmoid = 0.0f;
    if (x >= 0.0f) {
      float z = expf(-x);
      sigmoid = 1.0f / (1.0f + z);
    } else {
      float z = expf(x);
      sigmoid = z / (1.0f + z);
    }

    dataResult[idx] = sigmoid;
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

// NOTE: API Function

TensorMeta _TensorUnary_(Tensor A, Tensor &result) {
  auto storageA = A->storage;
  auto rankA = A->rank;
  auto shapeA = A->shape, stridesA = A->strides;

  // Logical number of elements of A
  size_t elementsA = calculateLogicalElementNo(shapeA, rankA);
  TensorMeta meta(rankA, elementsA, shapeA, stridesA);

  auto storageResult = std::make_shared<TensorStorage>(
      elementsA * sizeof(float), storageA->device);
  result = std::make_shared<TensorView>(shapeA, rankA);
  result->storage = storageResult;

  return meta;
}

Tensor TensorNeg(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorNeg(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorAbs(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorAbs(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorSqrt(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorSqrt(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}
Tensor TensorReciprocal(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorReciprocal(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorExp(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorExp(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorNLog(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorNLog(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorLog2(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorLog2(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}
Tensor TensorLog10(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorLog10(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorSin(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorSin(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorSinh(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorSinh(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}
Tensor TensorCos(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorCos(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}
Tensor TensorCosh(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorCosh(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}
Tensor TensorTan(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorTan(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}
Tensor TensorTanh(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorTanh(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}
Tensor TensorFloor(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorFloor(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}
Tensor TensorCeil(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorCeil(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorScale(Tensor A, float a) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorScale(dataA, meta, a, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorRelu(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorRelu(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorSigmoid(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorSigmoid(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}

Tensor TensorSoftmax(Tensor A) {
  Tensor result;
  auto meta = _TensorUnary_(A, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    // TODO: Optimization: Check if contiguous, in which case run the super
    // simple kernel - both fore CUDA and CPU
    CPU_TensorSoftmax(dataA, meta, result->storage->data);
    break;
  case StorageDevice::CUDA:
    // Call GPU Kernel
    break;
  }

  return result;
}
