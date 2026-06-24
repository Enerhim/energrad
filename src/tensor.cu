#include "../include/op.cuh"
#include "../include/tensor.cuh"

__global__ void tensor_add_kernel(const float *A, const float *B, float *C,
                                  size_t N, size_t ndim, const size_t *shape,
                                  const size_t *strides_A,
                                  const size_t *strides_B) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  size_t remaining = idx;

  size_t offsetA = 0;
  size_t offsetB = 0;

  for (int dim = ndim - 1; dim >= 0; --dim) {
    size_t coord = remaining % shape[dim];
    remaining /= shape[dim];

    offsetA += coord * strides_A[dim];
    offsetB += coord * strides_B[dim];
  }

  C[idx] = A[offsetA] + B[offsetB];
}

__global__ void tensor_accumulate_kernel(const float *A, float *B, size_t N) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    B[idx] += A[idx];
  }
}

TensorObject::TensorObject(const std::string &label,
                           const std::vector<size_t> &shape, bool hasGrad,
                           std::shared_ptr<TensorStorage> storage)
    : label(label), shape(shape), hasGrad(hasGrad), storage(storage) {

  // Construct Strides
  strides.resize(shape.size());
  size_t stride = 1;
  for (size_t i = shape.size() - 1; i > 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  strides[0] = stride;
}

TensorStorage::~TensorStorage() {
  if (kind == MemoryKind::Device) {
    if (data_ptr)
      cudaFreeAsync(data_ptr, ctx->stream);
    if (grad_ptr)
      cudaFreeAsync(grad_ptr, ctx->stream);
  } else {
    free(data_ptr);
    free(grad_ptr);
  }
}

size_t TensorObject::noElements() const {
  size_t n = 1;
  for (size_t s : shape)
    n *= s;
  return n;
}

size_t TensorObject::getSize() const {
  size_t n = 1;
  for (size_t s : shape)
    n *= s;
  return n * sizeof(float);
}

std::vector<float> TensorObject::hostBuffer() {
  std::vector<float> h(storage->_elements);
  cudaMemcpyAsync(h.data(), storage->data_ptr, storage->_size,
                  cudaMemcpyDeviceToHost, storage->ctx->stream);
  cudaStreamSynchronize(storage->ctx->stream);
  return h;
}

std::vector<float> TensorObject::hostGradBuffer() {
  if (storage->grad_ptr == nullptr || !hasGrad)
    return {};

  std::vector<float> h(storage->_elements);
  cudaMemcpyAsync(h.data(), storage->grad_ptr, storage->_size,
                  cudaMemcpyDeviceToHost, storage->ctx->stream);
  cudaStreamSynchronize(storage->ctx->stream);
  return h;
}

void TensorObject::setGrad(const std::vector<float> &data) {

  if (hasGrad) {

    if (storage->grad_ptr == nullptr)
      cudaMallocAsync(&storage->grad_ptr, storage->_size, storage->ctx->stream);

    cudaMemcpyAsync(storage->grad_ptr, data.data(), storage->_size,
                    cudaMemcpyHostToDevice, storage->ctx->stream);
  } else {
    throw std::runtime_error("hasGrad = false for this Tensor!\n");
  }
}

void TensorObject::zeroGrad() {

  if (hasGrad) {

    if (storage->grad_ptr == nullptr)
      cudaMallocAsync(&storage->grad_ptr, storage->_size, storage->ctx->stream);

    cudaMemsetAsync(storage->grad_ptr, 0.0f, storage->_size,
                    storage->ctx->stream);
  } else {
    throw std::runtime_error("hasGrad = false for this Tensor!\n");
  }
}

void TensorObject::accumulateGrad(float *top_gradient) {
  if (!hasGrad)
    return;

  if (storage->grad_ptr == nullptr) {
    cudaMallocAsync(&storage->grad_ptr, storage->_size, storage->ctx->stream);
    cudaMemsetAsync(storage->grad_ptr, 0, storage->_size, storage->ctx->stream);
  }

  uint N = storage->_elements;
  const uint BLOCK_SIZE = 32;
  uint blocks = CEIL_DIV(N, BLOCK_SIZE);

  tensor_accumulate_kernel<<<blocks, BLOCK_SIZE, 0, storage->ctx->stream>>>(
      top_gradient, storage->grad_ptr, N);
}

TensorStorage::TensorStorage(size_t elements, MemoryKind kind,
                             std::shared_ptr<CudaContext> ctx,
                             const std::vector<float> &data)
    : _elements(elements), _size(elements * sizeof(float)), kind(kind),
      ctx(ctx) {
  if (kind == MemoryKind::Device) {
    cudaMallocAsync(&data_ptr, _size, ctx->stream);

    if (!data.empty()) {
      if (data.size() != _elements) {
        throw std::runtime_error("Error: Data provided as argument to Tensor "
                                 "is neither empty nor correct size.\n");
      }
      cudaMemcpyAsync(data_ptr, data.data(), _size, cudaMemcpyHostToDevice,
                      ctx->stream);
    }
  }
}

Tensor operator+(const Tensor &a, const Tensor &b) {
  if (a->getShape() != b->getShape()) {
    throw std::runtime_error(
        "Error: Shapes do not match during + operation.\n");
  }

  auto ctx = a->getCudaContext();

  Tensor result = std::make_shared<TensorObject>(
      "", a->getShape(), a->hasGradient() || b->hasGradient(),
      std::make_shared<TensorStorage>(a->noElements(), MemoryKind::Device,
                                      a->getCudaContext(),
                                      std::vector<float>{}));

  auto op = std::make_shared<AddOp>();
  op->setParents({a, b});
  result->setOperation(op);

  uint N = a->noElements();
  const uint BLOCK_SIZE = 32;
  uint blocks = CEIL_DIV(N, BLOCK_SIZE);

  auto A_shape = a->getShape();
  auto A_strides = a->getStrides();
  auto B_strides = b->getStrides();

  size_t *shape_A, *strides_A, *strides_B;
  size_t shape_A_size = A_shape.size() * sizeof(float),
         strides_A_size = A_strides.size() * sizeof(float),
         strides_B_size = B_strides.size() * sizeof(float);

  cudaMallocAsync(&shape_A, shape_A_size, ctx->stream);
  cudaMallocAsync(&strides_A, strides_A_size, ctx->stream);
  cudaMallocAsync(&strides_B, strides_B_size, ctx->stream);

  cudaMemcpyAsync(shape_A, A_shape.data(), shape_A_size, cudaMemcpyHostToDevice,
                  ctx->stream);
  cudaMemcpyAsync(strides_A, A_strides.data(), strides_A_size,
                  cudaMemcpyHostToDevice, ctx->stream);
  cudaMemcpyAsync(strides_B, B_strides.data(), strides_B_size,
                  cudaMemcpyHostToDevice, ctx->stream);

  tensor_add_kernel<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
      a->storage->data_ptr, b->storage->data_ptr, result->storage->data_ptr, N,
      a->getShape().size(), shape_A, strides_A, strides_B);

  cudaFreeAsync(shape_A, ctx->stream);
  cudaFreeAsync(strides_A, ctx->stream);
  cudaFreeAsync(strides_B, ctx->stream);
  return result;
}

std::vector<size_t> checkBroadcastable(const std::vector<size_t> &src_shape,
                                       const std::vector<size_t> &src_strides,
                                       const std::vector<size_t> &dst_shape) {
  size_t src_dims = src_shape.size();
  size_t dst_dims = dst_shape.size();

  if (src_dims > dst_dims)
    throw std::runtime_error("Error: Cannot broadcast to smaller shape.\n");

  std::vector<size_t> new_strides(dst_dims, 0);

  for (int i = 0; i < dst_dims; i++) {
    int dst_idx = dst_dims - 1 - i;
    int src_idx = src_dims - 1 - i;

    if (src_idx >= 0) {
      size_t src_dim_sz = src_shape[src_idx];
      size_t dst_dim_sz = dst_shape[dst_idx];

      if (src_dim_sz == dst_dim_sz) {
        new_strides[dst_idx] = src_strides[src_idx];
      } else if (src_dim_sz == 1) {
        new_strides[dst_idx] = 0;
      } else {
        throw std::runtime_error(
            "Error: Shapes are either not broadcastable or are the same.\n");
      }
    } else {
      new_strides[dst_idx] = 0;
    }
  }

  return new_strides;
}

Tensor expand(const Tensor &a, const std::vector<size_t> &target_shape) {
  std::vector<size_t> new_strides =
      checkBroadcastable(a->getShape(), a->getStrides(), target_shape);

  auto ctx = a->getCudaContext();

  Tensor result = std::make_shared<TensorObject>(
      "", target_shape, a->hasGradient(), a->getStorage());

  auto op = std::make_shared<ExpandOp>();
  op->setParents({a});
  op->forward_ctx.saved_tensors.push_back(TensorW(result));
  result->setOperation(op);
  result->strides = new_strides;
  return result;
}

Tensor make_tensor(const std::string &label, const std::vector<size_t> &shape,
                   bool hasGrad, const std::vector<float> &data,
                   std::shared_ptr<CudaContext> ctx) {
  size_t allocated_length = 1;
  for (size_t s : shape) {
    allocated_length *= s;
  }
  return std::make_shared<TensorObject>(
      label, shape, hasGrad,
      std::make_shared<TensorStorage>(allocated_length, MemoryKind::Device, ctx,
                                      data));
}
