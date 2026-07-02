#include "op.cuh"
#include "tensor.cuh"

// CUDA Kernels

__global__ void tensor_expand_backward_kernel(const float *top_gradient,
                                              float *gradient, size_t ndim,
                                              size_t N, const size_t *Y_shape,
                                              const size_t *Y_strides) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t remaining = idx;

  size_t offset = 0;

  if (idx >= N)
    return;

  for (int dim = ndim - 1; dim >= 0; --dim) {
    size_t coord = remaining % Y_shape[dim];
    remaining /= Y_shape[dim];

    offset += coord * Y_strides[dim];
  }

  atomicAdd(&gradient[offset], top_gradient[idx]);
}

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

// Utils
std::vector<size_t *>
toDeviceShapeStrides(const std::vector<size_t> &shape,
                     const std::vector<size_t> &strides,
                     std::shared_ptr<CudaContext> cuda_ctx) {

  size_t *shape_ptr, *strides_ptr;
  size_t shape_ptr_size = sizeof(size_t) * shape.size();
  size_t strides_ptr_size = sizeof(size_t) * strides.size();
  cudaMallocAsync(&shape_ptr, shape_ptr_size, cuda_ctx->stream);
  cudaMallocAsync(&strides_ptr, strides_ptr_size, cuda_ctx->stream);
  cudaMemcpyAsync(shape_ptr, shape.data(), shape_ptr_size,
                  cudaMemcpyHostToDevice, cuda_ctx->stream);
  cudaMemcpyAsync(strides_ptr, strides.data(), strides_ptr_size,
                  cudaMemcpyHostToDevice, cuda_ctx->stream);

  return {shape_ptr, strides_ptr};
}

void freeShapeStrides(std::vector<size_t *> shape_strides_ptr,
                      std::shared_ptr<CudaContext> cuda_ctx) {
  cudaFreeAsync(shape_strides_ptr[0], cuda_ctx->stream);
  cudaFreeAsync(shape_strides_ptr[1], cuda_ctx->stream);
}

bool checkBroadcastable(const std::vector<size_t> &shape,
                        const std::vector<size_t> &new_shape) {
  size_t dim_a = shape.size();
  size_t dim_b = new_shape.size();

  if (dim_b < dim_a)
    return false;

  int i = static_cast<int>(dim_a) - 1;
  int j = static_cast<int>(dim_b) - 1;

  while (i >= 0) {
    if (shape[i] != new_shape[j] && shape[i] != 1) {
      return false;
    }
    --i, --j;
  }

  return true;
}

std::vector<size_t> broadcastStrides(std::vector<size_t> shape,
                                     const std::vector<size_t> &new_shape,
                                     std::vector<size_t> strides) {
  std::vector<size_t> new_strides = strides;

  // Padding shape to same size
  while (shape.size() < new_shape.size()) {
    shape.insert(shape.begin(), 1);
    new_strides.insert(new_strides.begin(), 0);
  }

  size_t N = shape.size();

  for (size_t i = 0; i < N; i++) {
    if (shape[i] == 1 && new_shape[i] != 1)
      new_strides[i] = 0;
  }

  return new_strides;
}

// Forward op overloads and fns
Tensor operator+(const Tensor &a, const Tensor &b) {

  if (a->getShape() != b->getShape()) {
    throw std::runtime_error(
        "Error: Tensor shapes do not match during + operation.\n");
  }

  auto op = std::make_shared<AddOp>();
  op->setParents({a, b});

  auto storage_a = a->getStorage();
  auto storage_b = b->getStorage();
  auto grad_ctx_a = a->getGradContext();
  auto grad_ctx_b = b->getGradContext();
  auto cuda_ctx = storage_a->getCudaContext();

  auto a_shape = a->getShape();

  auto shape_strides_a =
      toDeviceShapeStrides(a_shape, a->getStrides(), cuda_ctx);
  auto shape_strides_b =
      toDeviceShapeStrides(b->getShape(), b->getStrides(), cuda_ctx);

  Tensor result = tensor("", a_shape, {}, cuda_ctx, op,
                         grad_ctx_a->hasGrad || grad_ctx_b->hasGrad,
                         grad_ctx_a->delGrad && grad_ctx_b->delGrad);

  auto result_storage = result->getStorage();

  uint N = storage_a->getNumElements();
  const size_t BLOCK_SIZE = 32;
  const size_t BLOCKS = CEIL_DIV(N, BLOCK_SIZE);

  tensor_add_kernel<<<BLOCKS, BLOCK_SIZE, 0, cuda_ctx->stream>>>(
      storage_a->devicePtr(), storage_b->devicePtr(),
      result_storage->devicePtr(), N, a_shape.size(), shape_strides_a[0],
      shape_strides_a[1], shape_strides_b[1]);

  freeShapeStrides(shape_strides_a, cuda_ctx);
  freeShapeStrides(shape_strides_b, cuda_ctx);

  return result;
}

Tensor expand(const Tensor &A, const std::vector<size_t> &new_shape) {

  auto A_shape = A->getShape();
  if (!checkBroadcastable(A_shape, new_shape)) {
    throw std::runtime_error("Error: Can not broadcast to new shape..\n");
  }

  auto new_strides = broadcastStrides(A_shape, new_shape, A->getStrides());

  auto A_storage = A->getStorage();
  auto A_grad_ctx = A->getGradContext();
  auto cuda_context = A_storage->getCudaContext();

  auto op = std::make_shared<ExpandOp>();
  op->setParents({A});

  auto result = tensor("", new_shape, {}, cuda_context, op, A_grad_ctx->hasGrad,
                       A_grad_ctx->delGrad);
  result->storage = A_storage;
  result->strides = new_strides;

  op->forward_ctx.saved_tensors.push_back(A);
  op->forward_ctx.saved_tensors.push_back(result);

  return result;
}

// Backward pass fns

std::vector<Tensor> AddOp::backward(Tensor top_gradient) {
  return {top_gradient, top_gradient};
}

std::vector<Tensor> ExpandOp::backward(Tensor top_gradient) {

  auto X = forward_ctx.saved_tensors[0];
  auto Y = forward_ctx.saved_tensors[1];

  auto X_shape = X->getShape();
  auto dim_in = X_shape.size();
  auto storage_X = X->getStorage();

  auto storage_Y = Y->getStorage();
  uint N = storage_Y->getNumElements();

  auto shape_strides_Y = toDeviceShapeStrides(Y->getShape(), Y->getStrides(),
                                              storage_Y->getCudaContext());
  const size_t BLOCK_SIZE = 32;
  const size_t BLOCKS = CEIL_DIV(N, BLOCK_SIZE);

  Tensor gradient = tensor("", X_shape, {}, storage_X->getCudaContext(),
                           nullptr, false, false);

  tensor_expand_backward_kernel<<<BLOCKS, BLOCK_SIZE, 0,
                                  storage_X->getCudaContext()->stream>>>(
      top_gradient->getStorage()->devicePtr(),
      gradient->getStorage()->devicePtr(), dim_in, N, shape_strides_Y[0],
      shape_strides_Y[1]);

  return {};
}
