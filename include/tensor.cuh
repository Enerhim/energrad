#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

class TensorStorage;
class TensorView;

using Tensor = std::shared_ptr<TensorView>;
using StoragePtr = std::shared_ptr<TensorStorage>;

#define MAX_DIMS 10
enum class StorageDevice { CPU, CUDA };

class TensorView : public std::enable_shared_from_this<TensorView> {
public:
  TensorView(std::array<size_t, MAX_DIMS> s, size_t rank);
  size_t rank;
  std::array<size_t, MAX_DIMS> shape;
  std::array<size_t, MAX_DIMS> strides;
  std::shared_ptr<TensorStorage> storage;
};

class TensorStorage {
public:
  TensorStorage(size_t bytes, StorageDevice device);
  ~TensorStorage();
  size_t bytes;
  float *data = nullptr;
  StorageDevice device;
};

typedef struct TensorMeta {

  TensorMeta(int ndims, size_t elements, std::array<size_t, MAX_DIMS> shape,
             std::array<size_t, MAX_DIMS> strides)
      : ndims(ndims), elements(elements), shape(shape), strides(strides) {};
  TensorMeta();
  size_t ndims = 0;
  size_t elements = 0;
  std::array<size_t, MAX_DIMS> shape;
  std::array<size_t, MAX_DIMS> strides;
} TensorMeta;

// NOTE: USER API

// IO, Init

// TODO:
Tensor TensorLoadFile(std::string filename);
void TensorLoadFile(Tensor A, std::string filename);

Tensor TensorInit(std::array<size_t, MAX_DIMS> &shape, size_t rank, float a,
                  StorageDevice device, bool makeStorage);
Tensor TensorCreate(const std::vector<size_t> &shape, float a,
                    StorageDevice device);

void TensorLoadData(Tensor A, const float *data, size_t size_);

Tensor TensorOnes(const std::vector<size_t> &shape, StorageDevice device);
Tensor TensorZeros(const std::vector<size_t> &shape, StorageDevice device);
Tensor TensorRand(const std::vector<size_t> &shape, StorageDevice device,
                  bool normal = false, float min = 0.0f, float max = 1.0f);
Tensor TensorCopy(Tensor A, bool newStorage = false);

// TODO:
void TensorMoveDevice(Tensor A, StorageDevice device); // More of a utility

// Unary Operations
TensorMeta _TensorUnary_(Tensor A, Tensor &result);
Tensor TensorNeg(Tensor A);
Tensor TensorAbs(Tensor A);
Tensor TensorSqrt(Tensor A);
Tensor TensorReciprocal(Tensor A);
Tensor TensorExp(Tensor A);
Tensor TensorNLog(Tensor A);
Tensor TensorLog2(Tensor A);
Tensor TensorLog10(Tensor A);
Tensor TensorSin(Tensor A);
Tensor TensorSinh(Tensor A);
Tensor TensorCos(Tensor A);
Tensor TensorCosh(Tensor A);
Tensor TensorTan(Tensor A);
Tensor TensorTanh(Tensor A);
Tensor TensorFloor(Tensor A);
Tensor TensorCeil(Tensor A);
Tensor TensorScale(Tensor A, float a);
Tensor TensorRelu(Tensor A);
Tensor TensorSigmoid(Tensor A);
Tensor TensorSoftmax(Tensor A);

Tensor TensorAbsBackward(Tensor A);
Tensor TensorReluBackward(Tensor A);
Tensor TensorSoftmaxBackward(Tensor A);

Tensor TensorTranspose(Tensor A, size_t dim1, size_t dim2);
Tensor TensorBroadcast(Tensor A, std::array<size_t, MAX_DIMS> targetShape,
                       size_t rank);
Tensor TensorContiguous(Tensor A);

Tensor TensorSum(Tensor A, size_t dim, bool keepDims);
Tensor TensorMean(Tensor A, size_t dim, bool keepDims);
Tensor TensorMin(Tensor A, size_t dim, bool keepDims);
Tensor TensorMax(Tensor A, size_t dim, bool keepDims);
Tensor TensorNorm(Tensor A, size_t dim, bool keepDims);

// Binary Operations
Tensor TensorAdd(Tensor A, Tensor B);
Tensor TensorSub(Tensor A, Tensor B);
Tensor TensorMul(Tensor A, Tensor B);
Tensor TensorMatmul(Tensor A, Tensor B, bool AT = false, bool BT = false);

// TODO: Comparison ops
Tensor TensorEquals(Tensor A, Tensor B);
Tensor TensorLT(Tensor A, Tensor B);
Tensor TensorGT(Tensor A, Tensor B);

// NOTE: CPU Kernels

// Init
void CPU_TensorFill(float *dataA, float a, TensorMeta &metaA);
void CPU_TensorZeros(float *dataA, TensorMeta &metaA);
void CPU_TensorOnes(float *dataA, TensorMeta &metaA);

// Unary Ops
void CPU_TensorNeg(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorAbs(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorSqrt(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorReciprocal(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorExp(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorNLog(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorLog2(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorLog10(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorSin(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorCos(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorSinh(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorCosh(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorTanh(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorFloor(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorCeil(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorFloor(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorScale(float *dataA, TensorMeta &metaA, float scalar,
                     float *dataResult);
void CPU_TensorRelu(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorSigmoid(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorSoftmax(float *dataA, TensorMeta &metaA, float *dataResult);

void CPU_TensorAbsBackward(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorReluBackward(float *dataA, TensorMeta &metaA, float *dataResult);
void CPU_TensorSoftmaxBackward(float *dataA, TensorMeta &metaA,
                               float *dataResult);

void CPU_TensorSum(float *dataA, TensorMeta &mA, size_t dimk,
                   float *dataResult);
void CPU_TensorMean(float *dataA, TensorMeta &mA, size_t dimk,
                    float *dataResult);
void CPU_TensorMin(float *dataA, TensorMeta &mA, size_t dimk,
                   float *dataResult);
void CPU_TensorMax(float *dataA, TensorMeta &mA, size_t dimk,
                   float *dataResult);
void CPU_TensorNorm(float *dataA, TensorMeta &mA, size_t dimk,
                    float *dataResult);

// Binary Ops
void CPU_TensorAdd(float *dataA, TensorMeta &metaA, float *dataB,
                   TensorMeta &metaB, float *dataResult);
void CPU_TensorSub(float *dataA, TensorMeta &metaA, float *dataB,
                   TensorMeta &metaB, float *dataResult);
void CPU_TensorMul(float *dataA, TensorMeta &metaA, float *dataB,
                   TensorMeta &metaB, float *dataResult);
void CPU_TensorMatmulRR(float *dataA, TensorMeta &metaA, float *dataB,
                        TensorMeta &metaB, float *dataResult);
