#include "../include/op.cuh"

// AddOp

__global__ void tensor_backbroadcast(float *dX, const float *dY,
                                     size_t no_elements, size_t rank,
                                     const size_t *shape_X,
                                     const size_t *strides_X,
                                     const size_t *strides_Y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < no_elements) {
    int dx_flat_index = 0;
    int current_dy_index = i;

    for (int d = 0; d < rank; d++) {
      int coord_in_Y_space = current_dy_index / strides_Y[d];
      current_dy_index = current_dy_index % strides_Y[d];

      int coord_in_X_space = (shape_X[d] == 1) ? 0 : coord_in_Y_space;
      dx_flat_index += coord_in_X_space * strides_X[d];
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
  }

  float *dX;
  cudaMallocAsync(&dX, no_elements_old * sizeof(float), ctx->stream);

  const uint BLOCK_SIZE = 32;
  uint blocks = CEIL_DIV(no_elements, BLOCK_SIZE);
  tensor_backbroadcast<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
      dX, top_gradient, no_elements, prev_shape.size(), prev_shape.data(),
      prev_strides.data(), new_strides.data());

  cudaFreeAsync(a->getStorage()->grad_ptr, ctx->stream);
  a->getStorage()->_elements = no_elements_old;
  a->getStorage()->_size = no_elements_old * sizeof(float);
  a->accumulateGrad(dX);
}
