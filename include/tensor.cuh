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

struct Buffer {
  Buffer() : ptr(nullptr), kind(MemoryKind::Host) {};
  Buffer(float *ptr, MemoryKind kind) : ptr(ptr), kind(kind) {};
  float *ptr;
  MemoryKind kind;
};

struct Context {
  std::vector<TensorW> saved_tensors;
};

class TensorObject : public std::enable_shared_from_this<TensorObject> {

public:
  TensorObject(const std::string &label, const std::vector<size_t> &shape,
               bool hasGrad, const std::vector<float> &data,
               std::shared_ptr<CudaContext> ctx);
  ~TensorObject();

  std::shared_ptr<CudaContext> getCudaContext() const { return ctx; }

  Buffer deviceBuffer() const { return data_buffer; }
  Buffer deviceGradBuffer() const { return grad_buffer; }

  std::vector<float> hostBuffer();
  std::vector<float> hostGradBuffer();

  const std::vector<size_t> &getShape() const { return shape; }
  size_t getSize() const { return _size; }
  size_t noElements() const { return _elements; }
  bool hasGradient() const { return hasGrad; }
  void setLabel(const std::string &l) { label = l; }
  const std::string &getLabel() const { return label; }
  std::shared_ptr<Operation> getOperation() const { return parent_op; }

  void setGrad(const Buffer &buffer);
  void zeroGrad();
  void accumulateGrad(const Buffer &top_gradient);
  void setOperation(std::shared_ptr<Operation> op) { parent_op = op; }
  friend Tensor operator+(const Tensor &, const Tensor &);

private:
  std::vector<size_t> shape;
  std::vector<size_t> strides;
  size_t _size;
  size_t _elements;

  bool hasGrad;
  Buffer data_buffer = Buffer(nullptr, MemoryKind::Device);
  Buffer grad_buffer = Buffer(nullptr, MemoryKind::Device);

  std::shared_ptr<CudaContext> ctx;
  std::string label;
  std::shared_ptr<Operation> parent_op;
};

Tensor operator+(const Tensor &a, const Tensor &b);

Tensor make_tensor(const std::string &label, const std::vector<size_t> &shape,
                   bool hasGrad, const std::vector<float> &data,
                   std::shared_ptr<CudaContext> ctx);
