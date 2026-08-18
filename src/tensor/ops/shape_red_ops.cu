#include <tensor.cuh>
#include <utils.cuh>

// NOTE: CUDA Kernels

// NOTE: CPU Kernels

void CPU_TensorSum(float *dataA, TensorMeta &mA, size_t dimk,
                   float *dataResult) {
  // Thinking in terms of fibres..
  size_t ndim = mA.ndims;
  size_t stride_k = mA.strides[dimk];
  size_t K = mA.shape[dimk]; // Length of the fibre
  size_t outElements =
      mA.elements / K; // We want to walk over the output fibres and add the
                       // inputs that map to it...

  for (size_t idxOutput = 0; idxOutput < outElements; idxOutput++) {
    size_t offset = 0; // What is the coordinate of fibre in output space
    size_t remaining = idxOutput;

    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
      if (dim == static_cast<int>(dimk))
        continue;
      size_t dim_size = mA.shape[dim];
      size_t indice = remaining % dim_size;
      remaining /= dim_size;

      offset += indice * mA.strides[dim];
    }

    // We have the fibre coordinate linearly as `offset`
    float sum = 0.0f;
    for (size_t fibreIdx = 0; fibreIdx < K; fibreIdx++) {
      // Sum over all elements of the fibre coordinate in the input space.
      sum += dataA[fibreIdx * stride_k + offset];
    }
    dataResult[idxOutput] = sum;
  }
}

void CPU_TensorMean(float *dataA, TensorMeta &mA, size_t dimk,
                    float *dataResult) {
  // Thinking in terms of fibres..
  size_t ndim = mA.ndims;
  size_t stride_k = mA.strides[dimk];
  size_t K = mA.shape[dimk]; // Length of the fibre
  size_t outElements =
      mA.elements / K; // We want to walk over the output fibres and add the
                       // inputs that map to it...

  for (size_t idxOutput = 0; idxOutput < outElements; idxOutput++) {
    size_t offset = 0; // What is the coordinate of fibre in output space
    size_t remaining = idxOutput;

    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
      if (dim == static_cast<int>(dimk))
        continue;
      size_t dim_size = mA.shape[dim];
      size_t indice = remaining % dim_size;
      remaining /= dim_size;

      offset += indice * mA.strides[dim];
    }

    // We have the fibre coordinate linearly as `offset`
    float sum = 0.0f;
    for (size_t fibreIdx = 0; fibreIdx < K; fibreIdx++) {
      // Sum over all elements of the fibre coordinate in the input space.
      sum += dataA[fibreIdx * stride_k + offset];
    }
    dataResult[idxOutput] = sum / K;
  }
}

void CPU_TensorMin(float *dataA, TensorMeta &mA, size_t dimk,
                   float *dataResult) {
  // Thinking in terms of fibres..
  size_t ndim = mA.ndims;
  size_t stride_k = mA.strides[dimk];
  size_t K = mA.shape[dimk]; // Length of the fibre
  size_t outElements =
      mA.elements / K; // We want to walk over the output fibres and add the
                       // inputs that map to it...

  for (size_t idxOutput = 0; idxOutput < outElements; idxOutput++) {
    size_t offset = 0; // What is the coordinate of fibre in output space
    size_t remaining = idxOutput;

    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
      if (dim == static_cast<int>(dimk))
        continue;
      size_t dim_size = mA.shape[dim];
      size_t indice = remaining % dim_size;
      remaining /= dim_size;

      offset += indice * mA.strides[dim];
    }

    // We have the fibre coordinate linearly as `offset`
    float min_ = dataA[0 + offset];
    for (size_t fibreIdx = 0; fibreIdx < K; fibreIdx++) {
      // Sum over all elements of the fibre coordinate in the input space.
      if (dataA[fibreIdx * stride_k + offset] < min_)
        min_ = dataA[fibreIdx * stride_k + offset];
    }
    dataResult[idxOutput] = min_;
  }
}

void CPU_TensorMax(float *dataA, TensorMeta &mA, size_t dimk,
                   float *dataResult) {
  // Thinking in terms of fibres..
  size_t ndim = mA.ndims;
  size_t stride_k = mA.strides[dimk];
  size_t K = mA.shape[dimk]; // Length of the fibre
  size_t outElements =
      mA.elements / K; // We want to walk over the output fibres and add the
                       // inputs that map to it...

  for (size_t idxOutput = 0; idxOutput < outElements; idxOutput++) {
    size_t offset = 0; // What is the coordinate of fibre in output space
    size_t remaining = idxOutput;

    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
      if (dim == static_cast<int>(dimk))
        continue;
      size_t dim_size = mA.shape[dim];
      size_t indice = remaining % dim_size;
      remaining /= dim_size;

      offset += indice * mA.strides[dim];
    }

    // We have the fibre coordinate linearly as `offset`
    float max_ = dataA[0 + offset];
    for (size_t fibreIdx = 0; fibreIdx < K; fibreIdx++) {
      // Sum over all elements of the fibre coordinate in the input space.
      if (dataA[fibreIdx * stride_k + offset] > max_)
        max_ = dataA[fibreIdx * stride_k + offset];
    }
    dataResult[idxOutput] = max_;
  }
}

void CPU_TensorNorm(float *dataA, TensorMeta &mA, size_t dimk,
                    float *dataResult) {
  // Thinking in terms of fibres..
  size_t ndim = mA.ndims;
  size_t stride_k = mA.strides[dimk];
  size_t K = mA.shape[dimk]; // Length of the fibre
  size_t outElements =
      mA.elements / K; // We want to walk over the output fibres and add the
                       // inputs that map to it...

  for (size_t idxOutput = 0; idxOutput < outElements; idxOutput++) {
    size_t offset = 0; // What is the coordinate of fibre in output space
    size_t remaining = idxOutput;

    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
      if (dim == static_cast<int>(dimk))
        continue;
      size_t dim_size = mA.shape[dim];
      size_t indice = remaining % dim_size;
      remaining /= dim_size;

      offset += indice * mA.strides[dim];
    }

    // We have the fibre coordinate linearly as `offset`
    float sum_squares = 0.0f;
    for (size_t fibreIdx = 0; fibreIdx < K; fibreIdx++) {
      // Sum over all elements of the fibre coordinate in the input space.
      sum_squares = dataA[fibreIdx * stride_k + offset];
    }
    dataResult[idxOutput] = sqrtf(sum_squares);
  }
}

// NOTE: API Functions

Tensor TensorBroadcast(Tensor A, std::array<size_t, MAX_DIMS> targetShape,
                       size_t rank) {
  auto storageA = A->storage;
  size_t rankA = A->rank;

  auto result = TensorInit(targetShape, rankA, 0.0f, storageA->device, false);
  result->storage = storageA;

  std::array<size_t, MAX_DIMS> &stridesA = A->strides;
  std::array<size_t, MAX_DIMS> &stridesResult = result->strides;

  size_t zeroPadding = rank - rankA;
  std::fill(stridesResult.begin(), stridesResult.begin() + zeroPadding, 0);
  std::copy(stridesA.begin(), stridesA.begin() + rankA,
            stridesResult.begin() + zeroPadding);

  for (size_t i = zeroPadding; i < rank; i++) {
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

TensorMeta _TensorReduction_(Tensor A, size_t reducedDim, bool keepDim,
                             Tensor &result) {
  auto storageA = A->storage;
  auto rankA = A->rank;
  auto shapeA = A->shape, stridesA = A->strides;

  // Logical number of elements of A
  size_t elementsA = calculateLogicalElementNo(shapeA, rankA);
  TensorMeta meta(rankA, elementsA, shapeA, stridesA);

  // Output construction
  auto shapeOut = shapeA;
  auto rankOut = rankA;
  if (keepDim) {
    shapeOut[reducedDim] = 1;
  } else if (rankA > 1) {
    for (size_t d = reducedDim; d + 1 < rankA; d++)
      shapeOut[d] = shapeA[d + 1];
    shapeOut[rankA - 1] = 0;
    rankOut = rankA - 1;
  } else {
    shapeOut[0] = 1; // rank-1 input, fully reduced
  }

  auto elementsReduced = calculateLogicalElementNo(shapeOut, rankOut);

  auto storageResult = std::make_shared<TensorStorage>(
      elementsReduced * sizeof(float), storageA->device);
  result = std::make_shared<TensorView>(shapeOut, rankOut);
  result->storage = storageResult;

  return meta;
}

Tensor TensorSum(Tensor A, size_t dim, bool keepDim) {
  Tensor result;
  auto meta = _TensorReduction_(A, dim, keepDim, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    CPU_TensorSum(dataA, meta, dim, result->storage->data);
    break;
  case StorageDevice::CUDA:
    break;
  }

  return result;
}

Tensor TensorMean(Tensor A, size_t dim, bool keepDim) {
  Tensor result;
  auto meta = _TensorReduction_(A, dim, keepDim, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    CPU_TensorMean(dataA, meta, dim, result->storage->data);
    break;
  case StorageDevice::CUDA:
    break;
  }

  return result;
}

Tensor TensorMin(Tensor A, size_t dim, bool keepDim) {
  Tensor result;
  auto meta = _TensorReduction_(A, dim, keepDim, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    CPU_TensorMin(dataA, meta, dim, result->storage->data);
    break;
  case StorageDevice::CUDA:
    break;
  }

  return result;
}

Tensor TensorMax(Tensor A, size_t dim, bool keepDim) {
  Tensor result;
  auto meta = _TensorReduction_(A, dim, keepDim, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    CPU_TensorMax(dataA, meta, dim, result->storage->data);
    break;
  case StorageDevice::CUDA:
    break;
  }

  return result;
}

Tensor TensorNorm(Tensor A, size_t dim, bool keepDim) {
  Tensor result;
  auto meta = _TensorReduction_(A, dim, keepDim, result);
  auto storageA = A->storage;
  auto dataA = storageA->data;

  switch (storageA->device) {
  case StorageDevice::CPU:
    CPU_TensorNorm(dataA, meta, dim, result->storage->data);
    break;
  case StorageDevice::CUDA:
    break;
  }

  return result;
}
