#pragma once
#include <memory>
#include <string>
#include <vector>

enum class OpType {
  None,
  Add,
  Subtract,
  Multiply,
  Divide,
  Pow,
  MatMul,
  Relu,
  Sigmoid,
  Softmax
};

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  // Manual Creation
  Tensor(const std::string &label, uint rows, uint cols, bool grad, float fill);
  Tensor(const std::string &label, uint rows, uint cols, bool grad, uint min,
         uint max);
  Tensor(const std::string &label, uint rows, uint cols, bool grad,
         const std::vector<float> &data);
  // Op Creation
  Tensor(const std::string &label, OpType op,
         const std::vector<TensorPtr> &parents,
         const std::vector<float> &parameters, uint rows, uint cols, bool grad);
  ~Tensor();
  // Getters & Setters

  const std::string &label() const { return m_label; };
  void setLabel(const std::string &label) { m_label = label; };

  std::vector<float> toHost() const;
  const float *getDevicePtr() const { return d_values; };

  std::vector<float> toHostGrad() const;
  float *getDevicePtrGrad() { return d_grads; };

  OpType op() const { return m_op; }
  uint rows() const { return m_rows; };
  uint cols() const { return m_cols; };

  bool hasGrad() const { return gradients; };

  // Display to Cout

  void pShape() const;
  void pData() const;
  void pGrads() const;

  void cudaGradFill(float fill);

  void backward();

private:
  uint m_rows, m_cols;
  std::string m_label;
  OpType m_op;
  std::vector<TensorPtr> m_parents;
  float *d_values;
  float *d_grads;
  bool gradients;
};

TensorPtr operator+(const TensorPtr &a, const TensorPtr &b);
TensorPtr operator-(const TensorPtr &a, const TensorPtr &b);
TensorPtr operator*(const TensorPtr &a, const TensorPtr &b);
TensorPtr operator/(const TensorPtr &a, const TensorPtr &b);
TensorPtr tPow(const TensorPtr &a, float b);
