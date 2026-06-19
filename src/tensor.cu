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
                           const std::vector<float> &data,
                           std::shared_ptr<CudaContext> ctx)
    : label(label), shape(shape), hasGrad(hasGrad), ctx(ctx) {

  // Construct Strides
  strides.resize(shape.size());
  size_t stride = 1;
  for (size_t i = shape.size() - 1; i > 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  strides[0] = stride;
  _elements = strides[0] * shape[0];
  _size = _elements * sizeof(float);

  // Allocate memory on GPU
  cudaMallocAsync(&data_buffer.ptr, _size, ctx->stream);

  if (!data.empty()) {
    if (data.size() != _elements) {
      // std::cout << "_elements: " << _elements
      //           << " | data.size(): " << data.size() << std::endl;
      throw std::runtime_error("Error: Data provided as argument to Tensor "
                               "is neither empty nor correct size.\n");
    }
    cudaMemcpyAsync(data_buffer.ptr, data.data(), _size, cudaMemcpyHostToDevice,
                    ctx->stream);
  }
}

TensorObject::~TensorObject() {
  if (data_buffer.kind == MemoryKind::Device)
    cudaFreeAsync(data_buffer.ptr, ctx->stream);
  if (grad_buffer.kind == MemoryKind::Device)
    cudaFreeAsync(grad_buffer.ptr, ctx->stream);
}

std::vector<float> TensorObject::hostBuffer() {
  std::vector<float> h(_elements);
  cudaMemcpyAsync(h.data(), data_buffer.ptr, _size, cudaMemcpyDeviceToHost,
                  ctx->stream);
  cudaStreamSynchronize(ctx->stream);
  return h;
}

std::vector<float> TensorObject::hostGradBuffer() {
  if (grad_buffer.ptr == nullptr || !hasGrad)
    return {};

  std::vector<float> h(_elements);
  cudaMemcpyAsync(h.data(), grad_buffer.ptr, _size, cudaMemcpyDeviceToHost,
                  ctx->stream);
  cudaStreamSynchronize(ctx->stream);
  return h;
}

void TensorObject::setGrad(const Buffer &buffer) {

  if (hasGrad) {

    if (grad_buffer.ptr == nullptr)
      cudaMallocAsync(&grad_buffer.ptr, _size, ctx->stream);
    switch (buffer.kind) {
    case MemoryKind::Host:
      cudaMemcpyAsync(grad_buffer.ptr, buffer.ptr, _size,
                      cudaMemcpyHostToDevice, ctx->stream);
      break;
    case MemoryKind::Device:
      cudaMemcpyAsync(grad_buffer.ptr, buffer.ptr, _size,
                      cudaMemcpyDeviceToDevice, ctx->stream);
      break;
    }
  } else {
    throw std::runtime_error("hasGrad = false for this Tensor!\n");
  }
}

void TensorObject::zeroGrad() {

  if (hasGrad) {

    if (grad_buffer.ptr == nullptr)
      cudaMallocAsync(&grad_buffer.ptr, _size, ctx->stream);

    cudaMemsetAsync(grad_buffer.ptr, 0.0f, _size, ctx->stream);
  } else {
    throw std::runtime_error("hasGrad = false for this Tensor!\n");
  }
}

void TensorObject::accumulateGrad(const Buffer &top_gradient) {
  if (!hasGrad)
    return;

  if (grad_buffer.ptr == nullptr) {
    cudaMallocAsync(&grad_buffer.ptr, _size, ctx->stream);
    cudaMemsetAsync(grad_buffer.ptr, 0, _size, ctx->stream);
    grad_buffer.kind = MemoryKind::Device;
  }

  uint N = _elements;
  const uint BLOCK_SIZE = 32;
  uint blocks = CEIL_DIV(N, BLOCK_SIZE);

  tensor_accumulate_kernel<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
      top_gradient.ptr, grad_buffer.ptr, N);
}

Tensor operator+(const Tensor &a, const Tensor &b) {
  if (a->getShape() != b->getShape()) {
    throw std::runtime_error(
        "Error: Shapes do not match during + operation.\n");
  }

  auto ctx = a->getCudaContext();
  Tensor result = std::make_shared<TensorObject>(
      "", a->getShape(), a->hasGradient() || b->hasGradient(),
      std::vector<float>{}, ctx);

  auto op = std::make_shared<AddOp>();
  op->setParents({a, b});
  result->setOperation(op);

  uint N = a->noElements();
  const uint BLOCK_SIZE = 32;
  uint blocks = CEIL_DIV(N, BLOCK_SIZE);

  tensor_add_kernel<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
      a->deviceBuffer().ptr, b->deviceBuffer().ptr, result->deviceBuffer().ptr,
      N);
  return result;
}

Tensor make_tensor(const std::string &label, const std::vector<size_t> &shape,
                   bool hasGrad, const std::vector<float> &data,
                   std::shared_ptr<CudaContext> ctx) {
  return std::make_shared<TensorObject>(label, shape, hasGrad, data, ctx);
}
