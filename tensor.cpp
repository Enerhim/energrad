#include "tensor.hpp"
#include <iostream>

// Ops
TensorPtr operator+(const TensorPtr &a, const TensorPtr &b) {
  if (a->rows() != b->rows() || a->cols() != b->cols()) {
    throw std::runtime_error("Invalid sizes during + operation!");
    exit(-1);
  }
  bool grad = false;
  if (a->hasGrad() && b->hasGrad())
    grad = true;
  return std::make_shared<Tensor>("", OpType::Add, std::vector<TensorPtr>{a, b},
                                  std::vector<float>{}, a->rows(), a->cols(),
                                  grad);
}

TensorPtr operator-(const TensorPtr &a, const TensorPtr &b) {
  if (a->rows() != b->rows() || a->cols() != b->cols()) {
    throw std::runtime_error("Invalid sizes during - operation!");
    exit(-1);
  }
  bool grad = false;
  if (a->hasGrad() && b->hasGrad())
    grad = true;
  return std::make_shared<Tensor>(
      "", OpType::Subtract, std::vector<TensorPtr>{a, b}, std::vector<float>{},
      a->rows(), a->cols(), grad);
}

// Hadamard Product
TensorPtr operator*(const TensorPtr &a, const TensorPtr &b) {
  if (a->rows() != b->rows() || a->cols() != b->cols()) {
    throw std::runtime_error("Invalid sizes during hadamard multiplication!");
    exit(-1);
  }
  bool grad = false;
  if (a->hasGrad() && b->hasGrad())
    grad = true;
  return std::make_shared<Tensor>(
      "", OpType::Multiply, std::vector<TensorPtr>{a, b}, std::vector<float>{},
      a->rows(), a->cols(), grad);
}

// Hadamard Division
TensorPtr operator/(const TensorPtr &a, const TensorPtr &b) {
  if (a->rows() != b->rows() || a->cols() != b->cols()) {
    throw std::runtime_error("Invalid sizes during hadamard division!");
    exit(-1);
  }
  bool grad = false;
  if (a->hasGrad() && b->hasGrad())
    grad = true;
  return std::make_shared<Tensor>(
      "", OpType::Divide, std::vector<TensorPtr>{a, b}, std::vector<float>{},
      a->rows(), a->cols(), grad);
}

TensorPtr tPow(const TensorPtr &a, float b) {
  bool grad = false;
  if (a->hasGrad())
    grad = true;
  return std::make_shared<Tensor>("", OpType::Pow, std::vector<TensorPtr>{a},
                                  std::vector<float>{b}, a->rows(), a->cols(),
                                  grad);
}

// Print Statements

void Tensor::pShape() const {
  std::cout << m_label << " (" << m_rows << " x " << m_cols << ")";
}

void Tensor::pData() const {
  std::vector<float> data = toHost();

  pShape();
  std::cout << "\n[\n";
  for (int i = 0; i < m_rows; i++) {
    std::cout << "\t[ ";
    for (int j = 0; j < m_cols; j++) {
      std::cout << data[i * m_cols + j] << " ";
    }
    std::cout << "],\n";
  }
  std::cout << "]" << std::endl;
}

void Tensor::pGrads() const {
  std::vector<float> data = toHostGrad();

  pShape();
  std::cout << "\n[\n";
  for (int i = 0; i < m_rows; i++) {
    std::cout << "\t[ ";
    for (int j = 0; j < m_cols; j++) {
      std::cout << data[i * m_cols + j] << " ";
    }
    std::cout << "],\n";
  }
  std::cout << "]" << std::endl;
}
