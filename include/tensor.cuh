#pragma once
#include "cuda_context.cuh"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

enum class MemoryKind { Host, Device };
class Operation;
class TensorObject;
class Engine;

using Tensor = std::shared_ptr<TensorObject>;
using TensorW = std::weak_ptr<TensorObject>;

struct Context {
  std::vector<TensorW> saved_tensors;
};

struct TensorStorage {
  float *data_ptr = nullptr;
  float *grad_ptr = nullptr;
  size_t _size;
  size_t _elements;

  MemoryKind kind;
  std::shared_ptr<CudaContext> ctx;
  TensorStorage(size_t elements, MemoryKind kind,
                std::shared_ptr<CudaContext> ctx,
                const std::vector<float> &data);
  ~TensorStorage();
};

class TensorObject : public std::enable_shared_from_this<TensorObject> {

public:
  TensorObject(const std::string &label, const std::vector<size_t> &shape,
               bool hasGrad, std::shared_ptr<TensorStorage> storage);

  float *deviceData() const { return storage->data_ptr; };
  float *deviceGrad() const { return storage->grad_ptr; };

  std::vector<float> hostBuffer();
  std::vector<float> hostGradBuffer();

  const std::vector<size_t> &getShape() const { return shape; }
  const std::vector<size_t> &getStrides() const { return strides; }
  size_t getSize() const;
  size_t noElements() const;
  bool hasGradient() const { return hasGrad; }
  const std::string &getLabel() const { return label; }
  std::shared_ptr<Operation> getOperation() const { return parent_op; }
  std::shared_ptr<CudaContext> getCudaContext() const { return storage->ctx; }
  std::shared_ptr<TensorStorage> getStorage() const { return storage; }

  void setLabel(const std::string &l) { label = l; }
  void setGrad(const std::vector<float> &data);
  void zeroGrad();
  void accumulateGrad(float *top_grad);
  void setOperation(std::shared_ptr<Operation> op) { parent_op = op; }

  // Friend ops
  friend Tensor operator+(const Tensor &, const Tensor &);
  friend Tensor expand(const Tensor &, const std::vector<size_t> &);
  friend Tensor flatten(const Tensor &a);

private:
  std::vector<size_t> shape;
  std::vector<size_t> strides;
  std::shared_ptr<TensorStorage> storage;
  bool hasGrad;
  std::string label;
  std::shared_ptr<Operation> parent_op;
};

Tensor operator+(const Tensor &a, const Tensor &b);
Tensor expand(const Tensor &a, const std::vector<size_t> &shape);
Tensor flatten(const Tensor &a);

Tensor make_tensor(const std::string &label, const std::vector<size_t> &shape,
                   bool hasGrad, const std::vector<float> &data,
                   std::shared_ptr<CudaContext> ctx);
