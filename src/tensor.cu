#include "../include/tensor.cuh"
#include <stdexcept>

// CUDA Kernels
__global__ void tensor_accumulate_kernel(const float *A, float *B, size_t N) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    B[idx] += A[idx];
  }
}

// Grad Context

GradContext::GradContext(std::shared_ptr<Operation> parent_op, bool hasGrad,
                         bool delGrad)
    : op(parent_op), hasGrad(hasGrad), delGrad(delGrad) {

      };

// Tensor Storage
TensorStorage::TensorStorage(size_t allocation_size,
                             std::shared_ptr<CudaContext> cuda_ctx)
    : cuda_ctx(cuda_ctx) {

  _size = allocation_size;
  _elements = allocation_size / sizeof(float);

  cudaMallocAsync(&data_ptr, _size, cuda_ctx->stream);
  cudaMemsetAsync(data_ptr, 0, _size, cuda_ctx->stream);
}

TensorStorage::~TensorStorage() {
  if (data_ptr)
    cudaFreeAsync(data_ptr, cuda_ctx->stream);
}

void TensorStorage::setData(const std::vector<float> &data) {
  if ((data.size() * sizeof(float)) != _size)
    throw std::runtime_error("Error: Invalid size for setData.\n");

  cudaMemcpyAsync(data_ptr, data.data(), _size, cudaMemcpyHostToDevice,
                  cuda_ctx->stream);
}

// Tensr Object

TensorObject::TensorObject(const std::string &label,
                           const std::vector<size_t> &shape,
                           std::shared_ptr<TensorStorage> storage,
                           std::shared_ptr<GradContext> grad_ctx)
    : label(label), shape(shape), storage(storage), grad_ctx(grad_ctx) {

  // Check if size of storage = size of shapes
  size_t size_from_shape = sizeof(float);
  for (size_t s : shape)
    size_from_shape *= s;

  if (storage->getSize() != size_from_shape)
    throw std::runtime_error("Error: Mismatch between TensorObject size from "
                             "shape and TensorStorage size.\n");

  // Construct Strides
  strides.resize(shape.size());
  size_t stride = 1;
  for (size_t i = shape.size() - 1; i > 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  strides[0] = stride;
}

std::vector<float> TensorObject::toHost() {
  size_t ndims = shape.size();

  auto devicePtr = storage->devicePtr();
  auto cuda_ctx = storage->getCudaContext();
  auto _size = storage->getSize();
  auto _elements = storage->getNumElements();

  std::vector<float> t(_elements);
  cudaMemcpyAsync(t.data(), devicePtr, _size, cudaMemcpyDeviceToHost,
                  cuda_ctx->stream);
  cudaStreamSynchronize(cuda_ctx->stream);

  std::vector<float> data;
  std::vector<int> indices(ndims, 0);
  size_t increased_no_of_elements = 1;
  for (auto s : shape) {
    increased_no_of_elements *= s;
  }

  for (int i = 0; i < increased_no_of_elements; i++) {
    int offset = 0;

    for (int k = 0; k < ndims; k++) {
      offset += strides[k] * indices[k];
    }

    data.push_back(t[offset]);

    indices[ndims - 1]++;
    for (int k = ndims - 1; k > 0; k--) {
      if (indices[k] >= shape[k]) {
        indices[k] = 0;
        indices[k - 1]++;
      } else {
        break;
      }
    }
  }

  return data;
}

void allocateGrad(const std::vector<size_t> &shape, float fill,
                  std::shared_ptr<GradContext> grad_ctx,
                  std::shared_ptr<CudaContext> cuda_ctx) {

  size_t allocated_elements = 1;
  for (size_t s : shape) {
    allocated_elements *= s;
  }
  size_t allocated_size = allocated_elements * sizeof(float);
  // Lazily allocate
  if (!grad_ctx->grad) {
    grad_ctx->grad = tensor("", shape, {}, cuda_ctx, nullptr, false, false);
    // Tensor initialization memsets to 0 so not needed to do that again
    // auto grad_storage = grad_ctx->grad->getStorage();
    // cudaMemsetAsync(grad_storage->devicePtr(), 0, allocated_size,
    //                 cuda_ctx->stream);
  }
}

void TensorObject::accumulateGradient(
    Tensor top_gradient, std::shared_ptr<CudaContext> cuda_context) {
  if (!grad_ctx) {
    throw std::runtime_error("Error: Grad Context does not exist.\n");
  } else if (!grad_ctx->hasGrad) {
    throw std::runtime_error("Error: Tensor has `hasGrad` = false.\n");
  }

  if (!grad_ctx->grad)
    allocateGrad(shape, 0.0f, grad_ctx, cuda_context);
  auto grad_tensor = grad_ctx->grad;

  size_t allocated_elements = 1;
  for (size_t s : shape) {
    allocated_elements *= s;
  }
  auto top_grad_storage = top_gradient->getStorage();
  auto grad_storage = grad_tensor->getStorage();

  // Call accumulate kernel
  const size_t BLOCK_SIZE = 32;
  const size_t BLOCKS = CEIL_DIV(allocated_elements, BLOCK_SIZE);
  tensor_accumulate_kernel<<<BLOCKS, BLOCK_SIZE, 0, cuda_context->stream>>>(
      top_grad_storage->devicePtr(), grad_storage->devicePtr(),
      allocated_elements);
}

void TensorObject::freeGradient(std::shared_ptr<CudaContext> cuda_context) {
  if (!grad_ctx) {
    throw std::runtime_error("Error: Grad Context does not exist.\n");
  } else if (!grad_ctx->hasGrad) {
    throw std::runtime_error("Error: Tensor has `hasGrad` = false.\n");
  }

  if (grad_ctx->delGrad) {
    grad_ctx->grad = nullptr;
  }
}

// Make tensor
Tensor tensor(const std::string &label, const std::vector<size_t> &shape,
              const std::vector<float> &data, std::shared_ptr<CudaContext> ctx,
              std::shared_ptr<Operation> parent_op, bool hasGrad,
              bool delGrad) {

  size_t allocated_size = sizeof(float);
  for (size_t s : shape) {
    allocated_size *= s;
  }

  auto storage = std::make_shared<TensorStorage>(allocated_size, ctx);
  if (!data.empty())
    storage->setData(data);

  auto grad_ctx = std::make_shared<GradContext>(parent_op, hasGrad, delGrad);

  return std::make_shared<TensorObject>(label, shape, storage, grad_ctx);
}
