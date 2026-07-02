#include "../include/op.cuh"

// CUDA Kernels

__global__ void tensor_backbroadcast(float *dX, const float *dY,
                                     size_t no_elements, size_t rank,
                                     const size_t *shape_X,
                                     const size_t *strides_X,
                                     const size_t *shape_Y) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < no_elements) {
    size_t remaining = i;
    size_t dx_flat_index = 0;

    for (int d = rank - 1; d >= 0; --d) {
      size_t coord = remaining % shape_Y[d];
      remaining /= shape_Y[d];

      size_t coord_X = (shape_X[d] == 1) ? 0 : coord;
      dx_flat_index += coord_X * strides_X[d];
    }

    atomicAdd(&dX[dx_flat_index], dY[i]);
  }
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

// Forward op overloads and fns

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
  auto grad_ctx_b = a->getGradContext();
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
// Backward pass fns

std::vector<Tensor> AddOp::backward(Tensor top_gradient) {
  return {top_gradient, top_gradient};
}
