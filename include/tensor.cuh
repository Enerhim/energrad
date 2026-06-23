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
  float *data_ptr;
  float *grad_ptr;
  size_t _size;
  size_t _elements;

  MemoryKind kind;
  std::shared_ptr<CudaContext> ctx;

  ~TensorStorage();
};

class TensorObject : public std::enable_shared_from_this<TensorObject> {

public:
  TensorObject(const std::string &label, const std::vector<size_t> &shape,
               bool hasGrad, const std::vector<float> &data,
               std::shared_ptr<TensorStorage> storage);

  std::vector<float> deviceData() const {
    return std::vector<float>(storage->data_ptr,
                              storage->data_ptr + storage->_size);
  };
  std::vector<float> deviceGrad() const {
    return std::vector<float>(storage->grad_ptr,
                              storage->grad_ptr + storage->_size);
  };

  std::vector<float> hostBuffer();
  std::vector<float> hostGradBuffer();

  const std::vector<size_t> &getShape() const { return shape; }
  size_t getSize() const { return storage->_size; }
  size_t noElements() const { return storage->_elements; }
  bool hasGradient() const { return hasGrad; }
  const std::string &getLabel() const { return label; }
  std::shared_ptr<Operation> getOperation() const { return parent_op; }
  std::shared_ptr<CudaContext> getCudaContext() const { return storage->ctx; }

  void setLabel(const std::string &l) { label = l; }
  void setGrad(const std::vector<float> &data);
  void zeroGrad();
  void accumulateGrad(const std::vector<float> &top_grad);
  void setOperation(std::shared_ptr<Operation> op) { parent_op = op; }
  friend Tensor operator+(const Tensor &, const Tensor &);

private:
  std::vector<size_t> shape;
  std::vector<size_t> strides;

  std::shared_ptr<TensorStorage> storage;

  bool hasGrad;

  std::string label;
  std::shared_ptr<Operation> parent_op;
};

Tensor operator+(const Tensor &a, const Tensor &b);

Tensor make_tensor(const std::string &label, const std::vector<size_t> &shape,
                   bool hasGrad, const std::vector<float> &data,
                   std::shared_ptr<CudaContext> ctx);
