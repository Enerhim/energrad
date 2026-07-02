#pragma once
#include "context.cuh"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

class Operation;
class TensorObject;
class GradientMetadata;
class Engine;

using Tensor = std::shared_ptr<TensorObject>;
using TensorW = std::weak_ptr<TensorObject>;

class GradContext {
public:
  GradContext(std::shared_ptr<Operation> parent_op, bool hasGrad, bool delGrad);

  Tensor grad = nullptr;
  std::shared_ptr<Operation> op = nullptr;

  bool hasGrad = false;
  bool delGrad = false;
};

class TensorStorage : public std::enable_shared_from_this<TensorStorage> {
private:
  float *data_ptr = nullptr;
  size_t _size;
  size_t _elements;
  std::shared_ptr<CudaContext> cuda_ctx;

  void setData(const std::vector<float> &data);

public:
  TensorStorage(size_t allocation_size, std::shared_ptr<CudaContext> ctx);
  ~TensorStorage();

  float *devicePtr() const { return data_ptr; }
  size_t getSize() const { return _size; }
  size_t getNumElements() const { return _elements; }

  std::shared_ptr<CudaContext> getCudaContext() const { return cuda_ctx; }

  friend Engine;
  friend Tensor tensor(const std::string &, const std::vector<size_t> &,
                       const std::vector<float> &, std::shared_ptr<CudaContext>,
                       std::shared_ptr<Operation>, bool, bool);
};

class TensorObject : public std::enable_shared_from_this<TensorObject> {

public:
  TensorObject(const std::string &label, const std::vector<size_t> &shape,
               std::shared_ptr<TensorStorage> storage,
               std::shared_ptr<GradContext> grad_ctx);

  // Getters
  const std::vector<size_t> &getShape() const { return shape; }
  const std::vector<size_t> &getStrides() const { return strides; }

  std::shared_ptr<GradContext> getGradContext() const { return grad_ctx; }
  std::shared_ptr<TensorStorage> getStorage() const { return storage; }

  const std::string &getLabel() const { return label; }

  std::vector<float> toHost();

  // Setters
  void setLabel(const std::string &l) { label = l; }

  // Backprop
  void accumulateGradient(Tensor gradient,
                          std::shared_ptr<CudaContext> cuda_context);
  void freeGradient(std::shared_ptr<CudaContext> cuda_context);

  // Friend ops
  friend Tensor operator+(const Tensor &, const Tensor &);
  friend Tensor expand(const Tensor &, const std::vector<size_t> &);
  friend Tensor flatten(const Tensor &a);

private:
  std::vector<size_t> shape;
  std::vector<size_t> strides;
  std::shared_ptr<TensorStorage> storage;
  std::shared_ptr<GradContext> grad_ctx;
  std::string label;
};
void allocateGrad(const std::vector<size_t> &shape, float fill,
                  std::shared_ptr<GradContext> grad_ctx,
                  std::shared_ptr<CudaContext> cuda_ctx);
Tensor tensor(const std::string &label, const std::vector<size_t> &shape,
              const std::vector<float> &data, std::shared_ptr<CudaContext> ctx,
              std::shared_ptr<Operation> parent_op, bool hasGrad, bool delGrad);
