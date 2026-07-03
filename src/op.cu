#include "op.cuh"
#include "tensor.cuh"
#include <stdexcept>
// CUDA Kernels

__global__ void tensor_expand_backward_kernel(const float *dY, float *dX,
                                              size_t ndim, size_t N,
                                              const size_t *Y_shape,
                                              const size_t *dY_strides,
                                              const size_t *dX_strides) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N)
    return;

  size_t remaining = idx;

  size_t offset_dY = 0, offset_dX = 0;
  for (int dim = ndim - 1; dim >= 0; --dim) {
    size_t coord = remaining % Y_shape[dim];
    remaining /= Y_shape[dim];

    offset_dX += coord * dX_strides[dim];
    offset_dY += coord * dY_strides[dim];
  }

  atomicAdd(&dX[offset_dX], dY[offset_dY]);
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

  size_t N = storage_a->getNumElements();
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
  result->setStrides(new_strides);
  result->setStorage(A_storage);

  op->forward_ctx.saved_tensors.push_back(A);
  op->forward_ctx.saved_tensors.push_back(result);

  return result;
}

Tensor transpose(const Tensor &A, size_t dim_i, size_t dim_j) {
  auto A_shape = A->getShape();
  auto A_strides = A->getStrides();
  auto A_storage = A->getStorage();
  auto A_grad_ctx = A->getGradContext();
  auto cuda_context = A_storage->getCudaContext();
  auto n_dims = A_shape.size();
  if (dim_i >= n_dims || dim_j >= n_dims)
    throw std::runtime_error("Error: Transpose dims out of bounds\n.");

  std::swap(A_shape[dim_i], A_shape[dim_j]);
  std::swap(A_strides[dim_i], A_strides[dim_j]);

  auto op = std::make_shared<TransposeOp>();
  op->setParents({A});

  auto result = tensor("", A_shape, {}, cuda_context, op, A_grad_ctx->hasGrad,
                       A_grad_ctx->delGrad);

  result->setStrides(A_strides);
  result->setStorage(A_storage);

  op->forward_ctx.saved_floats.push_back(dim_i);
  op->forward_ctx.saved_floats.push_back(dim_j);

  return result;
}

// Backward pass fns

std::vector<Tensor> AddOp::backward(Tensor top_gradient) {
  return {top_gradient, top_gradient};
}

// TODO: Reanalyze and rewrite step by step to understand this again thoroughly.
std::vector<Tensor> ExpandOp::backward(Tensor dY) {
  auto X = forward_ctx.saved_tensors[0];
  auto Y = forward_ctx.saved_tensors[1];

  auto X_shape = X->getShape();
  auto Y_shape = Y->getShape();
  auto X_storage = X->getStorage();
  auto cuda_ctx_X = X_storage->getCudaContext();

  size_t N = 1;
  for (auto s : Y_shape)
    N *= s;
  size_t ndim = Y_shape.size();
  size_t BLOCK_SIZE = 32;
  size_t BLOCKS = CEIL_DIV(N, BLOCK_SIZE);

  auto dX = tensor("", X_shape, {}, cuda_ctx_X, nullptr, false, true);

  std::vector<size_t> X_contiguous_strides(X_shape.size());
  size_t stride = 1;
  for (int i = static_cast<int>(X_shape.size()) - 1; i >= 0; i--) {
    X_contiguous_strides[i] = stride;
    stride *= X_shape[i];
  }

  auto dX_m_strides = broadcastStrides(X_shape, Y_shape, X_contiguous_strides);

  auto Y_dX_device = toDeviceShapeStrides(Y_shape, dX_m_strides, cuda_ctx_X);
  auto Y_dY_device =
      toDeviceShapeStrides(Y_shape, dY->getStrides(), cuda_ctx_X);
  auto dY_ptr = dY->getStorage()->devicePtr();
  auto dX_ptr = dX->getStorage()->devicePtr();
  tensor_expand_backward_kernel<<<BLOCKS, BLOCK_SIZE, 0, cuda_ctx_X->stream>>>(
      dY_ptr, dX_ptr, ndim, N, Y_dY_device[0], Y_dY_device[1], Y_dX_device[1]);

  freeShapeStrides(Y_dX_device, cuda_ctx_X);
  freeShapeStrides(Y_dY_device, cuda_ctx_X);

  return {dX};
}

std::vector<Tensor> TransposeOp::backward(Tensor dY) {
  auto dY_shape = dY->getShape();
  auto dY_strides = dY->getStrides();
  auto dY_storage = dY->getStorage();

  auto dim_i = forward_ctx.saved_floats[0];
  auto dim_j = forward_ctx.saved_floats[1];

  auto cuda_context = dY_storage->getCudaContext();
  auto n_dims = dY_shape.size();

  std::swap(dY_shape[dim_i], dY_shape[dim_j]);
  std::swap(dY_strides[dim_i], dY_strides[dim_j]);

  auto result = tensor("", dY_shape, {}, cuda_context, nullptr, false, true);

  result->setStrides(dY_strides);
  result->setStorage(dY_storage);
  return {result};
}
