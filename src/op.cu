#include "../include/op.cuh"

// TODO: Optimize with blocking m8
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

void AddOp::backward(float *top_gradient) {
  auto a = parents[0].lock();
  auto b = parents[1].lock();

  a->accumulateGrad(top_gradient);
  b->accumulateGrad(top_gradient);
}

void ExpandOp::backward(float *top_gradient) {
  auto a = parents[0].lock();
  auto a_ = forward_ctx.saved_tensors[0].lock();

  std::vector<size_t> prev_shape = a->getShape();
  std::vector<size_t> new_shape = a_->getShape();
  std::vector<size_t> prev_strides = a->getStrides();
  std::vector<size_t> new_strides = a_->getStrides();
  auto ctx = a->getCudaContext();

  size_t no_elements = 1;
  for (size_t size_ : new_shape)
    no_elements *= size_;

  size_t no_elements_old = 1;
  for (size_t size_ : prev_shape)
    no_elements_old *= size_;

  while (new_shape.size() > prev_shape.size()) {
    prev_shape.insert(prev_shape.begin(), 1);
    prev_strides.insert(prev_strides.begin(), 0);
  }
  float *dX;
  cudaMallocAsync(&dX, no_elements_old * sizeof(size_t), ctx->stream);
  cudaMemsetAsync(dX, 0, no_elements_old * sizeof(size_t), ctx->stream);

  size_t *shape_X, *strides_X, *shape_Y;
  size_t shape_X_size = prev_shape.size() * sizeof(size_t),
         strides_X_size = prev_strides.size() * sizeof(size_t),
         shape_Y_size = new_shape.size() * sizeof(size_t);

  cudaMallocAsync(&shape_X, shape_X_size, ctx->stream);
  cudaMallocAsync(&strides_X, strides_X_size, ctx->stream);
  cudaMallocAsync(&shape_Y, shape_Y_size, ctx->stream);

  cudaMemcpyAsync(shape_X, prev_shape.data(), shape_X_size,
                  cudaMemcpyHostToDevice, ctx->stream);
  cudaMemcpyAsync(strides_X, prev_strides.data(), strides_X_size,
                  cudaMemcpyHostToDevice, ctx->stream);
  cudaMemcpyAsync(shape_Y, new_strides.data(), shape_Y_size,
                  cudaMemcpyHostToDevice, ctx->stream);

  const uint BLOCK_SIZE = 32;
  uint blocks = CEIL_DIV(no_elements, BLOCK_SIZE);
  tensor_backbroadcast<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
      dX, top_gradient, no_elements, prev_shape.size(), shape_X, strides_X,
      shape_Y);

  cudaFreeAsync(shape_X, ctx->stream);
  cudaFreeAsync(strides_X, ctx->stream);
  cudaFreeAsync(shape_Y, ctx->stream);

  a->accumulateGrad(dX);
}
