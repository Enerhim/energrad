#pragma once
#include "tensor.cuh"

class Operation {
public:
  virtual void backward(const Buffer &top_gradient) = 0;
  virtual ~Operation() = default;

  const std::vector<TensorW> &getParents() const { return parents; }
  friend Tensor operator+(const Tensor &, const Tensor &);

protected:
  Context ctx;
  std::vector<TensorW> parents;
  void setParents(const std::vector<TensorW> &p) { parents = p; }
};

class AddOp : public Operation {
public:
  void backward(const Buffer &top_gradient) override;
};

class MatmulOp : public Operation {
public:
  void backward(const Buffer &top_gradient) override;
};
