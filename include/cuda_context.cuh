#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct CudaContext {
  int device = 0;
  cudaStream_t stream{};

  explicit CudaContext(int device = 0) : device(device) {
    cudaSetDevice(device);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, device);

    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
  }

  ~CudaContext() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }
};
