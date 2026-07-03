#pragma once
#include "tensor.cuh"

Tensor operator+(const Tensor &a, const Tensor &b);
Tensor expand(const Tensor &a, const std::vector<size_t> &shape);
Tensor flatten(const Tensor &a);
Tensor transpose(const Tensor &a, size_t dim_i, size_t dim_j);

struct BackpropContext {
  std::vector<Tensor> saved_tensors;
  std::vector<float> saved_floats;
};

class Operation {
public:
  virtual std::vector<Tensor> backward(Tensor top_gradient) = 0;
  virtual ~Operation() = default;

  friend Tensor operator+(const Tensor &, const Tensor &);
  friend Tensor expand(const Tensor &, const std::vector<size_t> &);
  friend Tensor flatten(const Tensor &);
  friend Tensor transpose(const Tensor &, size_t, size_t);

  virtual const std::vector<Tensor> &getParents() const { return parents; }

protected:
  std::vector<Tensor> parents;

  BackpropContext forward_ctx;
  void setParents(const std::vector<Tensor> &p) { parents = p; }
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

class TransposeOp : public Operation {
public:
  std::vector<Tensor> backward(Tensor top_gradient) override;
};

class ContiguousOp : public Operation {
public:
  std::vector<Tensor> backward(Tensor top_gradient) override;
};
