#pragma once
#include "tensor.cuh"

Tensor operator+(const Tensor &a, const Tensor &b);
Tensor expand(const Tensor &a, const std::vector<size_t> &shape);
Tensor flatten(const Tensor &a);

struct BackpropContext {
  std::vector<TensorW> saved_tensors;
};

class Operation {
public:
  virtual std::vector<Tensor> backward(Tensor top_gradient) = 0;
  virtual ~Operation() = default;

  friend Tensor operator+(const Tensor &, const Tensor &);
  friend Tensor expand(const Tensor &, const std::vector<size_t> &);
  friend Tensor flatten(const Tensor &);

  virtual const std::vector<TensorW> &getParents() const { return parents; }

protected:
  BackpropContext forward_ctx;
  std::vector<TensorW> parents;

  void setParents(const std::vector<TensorW> &p) { parents = p; }
};

// Operation Classes

class AddOp : public Operation {
public:
  std::vector<Tensor> backward(Tensor top_gradient) override;
};

class ExpandOp : public Operation {
public:
  std::vector<Tensor> backward(Tensor top_gradient) override;
};

class ContiguousOp : public Operation {
public:
  std::vector<Tensor> backward(Tensor top_gradient) override;
};
