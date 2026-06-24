#pragma once
#include "tensor.cuh"

class Operation {
public:
  virtual void backward(float *top_gradient) = 0;
  virtual ~Operation() = default;

  const std::vector<TensorW> &getParents() const { return parents; }
  friend Tensor operator+(const Tensor &, const Tensor &);
  friend Tensor expand(const Tensor &, const std::vector<size_t> &);
  friend Tensor flatten(const Tensor &);

protected:
  Context forward_ctx;
  std::vector<TensorW> parents;
  void setParents(const std::vector<TensorW> &p) { parents = p; }
};

class AddOp : public Operation {
public:
  void backward(float *top_gradient) override;
};

// Broadcasting and contiguousness

class ExpandOp : public Operation {
public:
  void backward(float *top_gradient) override;
};

class ContiguousOp : public Operation {
public:
  void backward(float *top_gradient) override;
};
