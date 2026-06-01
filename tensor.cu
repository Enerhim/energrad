#include "tensor.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>

#define BLOCK_SIZE 32
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
// Cuda Kernels

__global__ void cuda_mat_add_kernel(const float *A, const float *B, float *C,
                                    uint M, uint N) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < N && y < M) {
    C[y * N + x] = A[y * N + x] + B[y * N + x];
  }
}

__global__ void cuda_mat_sub_kernel(const float *A, const float *B, float *C,
                                    uint M, uint N) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < N && y < M) {
    C[y * N + x] = A[y * N + x] - B[y * N + x];
  }
}

__global__ void cuda_mat_hadamard_mul_kernel(const float *A, const float *B,
                                             float *C, uint M, uint N) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < N && y < M) {
    C[y * N + x] = A[y * N + x] * B[y * N + x];
  }
}

__global__ void cuda_mat_hadamard_div_kernel(const float *A, const float *B,
                                             float *C, uint M, uint N) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < N && y < M) {
    C[y * N + x] = (B[y * N + x] == 0) ? 0 : A[y * N + x] / B[y * N + x];
  }
}

__global__ void cuda_mat_pow_kernel(const float *A, float B, float *C, uint M,
                                    uint N) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < N && y < M) {
    C[y * N + x] = powf(A[y * N + x], B);
  }
}

// API

void cuda_mat_add(const std::vector<TensorPtr> &parents, float *C) {
  const float *A = parents[0]->getDevicePtr();
  const float *B = parents[1]->getDevicePtr();
  uint M = parents[0]->rows();
  uint N = parents[0]->cols();

  dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

  cuda_mat_add_kernel<<<gridDim, blockDim>>>(A, B, C, M, N);
  cudaDeviceSynchronize();
}

void cuda_mat_sub(const std::vector<TensorPtr> &parents, float *C) {
  const float *A = parents[0]->getDevicePtr();
  const float *B = parents[1]->getDevicePtr();
  uint M = parents[0]->rows();
  uint N = parents[0]->cols();

  dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

  cuda_mat_sub_kernel<<<gridDim, blockDim>>>(A, B, C, M, N);
  cudaDeviceSynchronize();
}

void cuda_mat_hadamard_mul(const std::vector<TensorPtr> &parents, float *C) {
  const float *A = parents[0]->getDevicePtr();
  const float *B = parents[1]->getDevicePtr();
  uint M = parents[0]->rows();
  uint N = parents[0]->cols();

  dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

  cuda_mat_hadamard_mul_kernel<<<gridDim, blockDim>>>(A, B, C, M, N);
  cudaDeviceSynchronize();
}

void cuda_mat_hadamard_div(const std::vector<TensorPtr> &parents, float *C) {
  const float *A = parents[0]->getDevicePtr();
  const float *B = parents[1]->getDevicePtr();
  uint M = parents[0]->rows();
  uint N = parents[0]->cols();

  dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

  cuda_mat_hadamard_div_kernel<<<gridDim, blockDim>>>(A, B, C, M, N);
  cudaDeviceSynchronize();
}

void cuda_mat_pow(const std::vector<TensorPtr> &parents, float b, float *C) {
  const float *A = parents[0]->getDevicePtr();
  uint M = parents[0]->rows();
  uint N = parents[0]->cols();

  dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

  cuda_mat_pow_kernel<<<gridDim, blockDim>>>(A, b, C, M, N);
  cudaDeviceSynchronize();
}

// Some Functions

std::vector<float> Tensor::toHost() const {
  uint size = m_rows * m_cols;
  std::vector<float> vals(size);

  cudaMemcpy(vals.data(), d_values, size * sizeof(float),
             cudaMemcpyDeviceToHost);

  return vals;
}

// Tensor Constructing

Tensor::Tensor(const std::string &label, uint rows, uint cols, bool grad,
               float fill)
    : m_label(label), m_rows(rows), m_cols(cols), m_op(OpType::None) {

  uint size = m_rows * m_cols;
  float *h_values = new float[size];
  for (int i = 0; i < size; i++)
    h_values[i] = fill;

  cudaMalloc(&d_values, size * sizeof(float));
  cudaMemcpy(d_values, h_values, size * sizeof(float), cudaMemcpyHostToDevice);
  delete[] h_values;
}

Tensor::Tensor(const std::string &label, uint rows, uint cols, bool grad,
               uint min, uint max)
    : m_label(label), m_rows(rows), m_cols(cols), m_op(OpType::None) {

  uint size = m_rows * m_cols;
  float *h_values = new float[size];
  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(min, max);

  for (int i = 0; i < size; i++)
    h_values[i] = dist(gen);

  cudaMalloc(&d_values, size * sizeof(float));
  cudaMemcpy(d_values, h_values, size * sizeof(float), cudaMemcpyHostToDevice);
  delete[] h_values;
}
Tensor::Tensor(const std::string &label, uint rows, uint cols, bool grad,
               const std::vector<float> &data)
    : m_label(label), m_rows(rows), m_cols(cols), m_op(OpType::None) {

  uint size = m_rows * m_cols;
  float *h_values = new float[size];

  if (data.size() == size) {
    for (int i = 0; i < size; i++)
      h_values[i] = data[i];
  } else {
    throw std::invalid_argument("Wrong size of data argument for tensor C: " +
                                m_label);
    exit(-1);
  }

  cudaMalloc(&d_values, size * sizeof(float));
  cudaMemcpy(d_values, h_values, size * sizeof(float), cudaMemcpyHostToDevice);
  delete[] h_values;
}

// Op Creation
Tensor::Tensor(const std::string &label, OpType op,
               const std::vector<TensorPtr> &parents,
               const std::vector<float> &parameters, uint rows, uint cols,
               bool grad)
    : m_label(label), m_rows(rows), m_cols(cols), m_op(op), m_parents(parents) {

  uint size = m_rows * m_cols;
  cudaMalloc(&d_values, size * sizeof(float));

  switch (op) {
  case OpType::Add:
    cuda_mat_add(parents, d_values);
    break;
  case OpType::Subtract:
    cuda_mat_sub(parents, d_values);
    break;
  case OpType::Multiply:
    cuda_mat_hadamard_mul(parents, d_values);
    break;
  case OpType::Divide:
    cuda_mat_hadamard_div(parents, d_values);
    break;
  case OpType::Pow:
    cuda_mat_pow(parents, parameters[0], d_values);
    break;
  }
}

Tensor::~Tensor() { cudaFree(d_values); }
