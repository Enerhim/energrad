#include "../include/op.cuh"
#include "../include/tensor.cuh"

__global__ void tensor_add_kernel(const float *A, const float *B, float *C,
                                  size_t N) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
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

  tensor_add_kernel<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
      a->storage->data_ptr, b->storage->data_ptr, result->storage->data_ptr, N);
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
