#include "tensor.hpp"
#include <iostream>

// Ops
TensorPtr operator+(const TensorPtr &a, const TensorPtr &b) {
  if (a->rows() != b->rows() || a->cols() != b->cols()) {
    throw std::runtime_error("Invalid sizes during + operation!");
    exit(-1);
  }
  return std::make_shared<Tensor>("", OpType::Add, std::vector<TensorPtr>{a, b},
                                  std::vector<float>{}, a->rows(), a->cols(),
                                  false);
}

TensorPtr operator-(const TensorPtr &a, const TensorPtr &b) {
  if (a->rows() != b->rows() || a->cols() != b->cols()) {
    throw std::runtime_error("Invalid sizes during - operation!");
    exit(-1);
  }
  return std::make_shared<Tensor>(
      "", OpType::Subtract, std::vector<TensorPtr>{a, b}, std::vector<float>{},
      a->rows(), a->cols(), false);
}

// Hadamard Product
TensorPtr operator*(const TensorPtr &a, const TensorPtr &b) {
  if (a->rows() != b->rows() || a->cols() != b->cols()) {
    throw std::runtime_error("Invalid sizes during hadamard multiplication!");
    exit(-1);
  }
  return std::make_shared<Tensor>(
      "", OpType::Multiply, std::vector<TensorPtr>{a, b}, std::vector<float>{},
      a->rows(), a->cols(), false);
}

// Hadamard Division
TensorPtr operator/(const TensorPtr &a, const TensorPtr &b) {
  if (a->rows() != b->rows() || a->cols() != b->cols()) {
    throw std::runtime_error("Invalid sizes during hadamard division!");
    exit(-1);
  }
  return std::make_shared<Tensor>(
      "", OpType::Divide, std::vector<TensorPtr>{a, b}, std::vector<float>{},
      a->rows(), a->cols(), false);
}

TensorPtr tPow(const TensorPtr &a, float b) {
  return std::make_shared<Tensor>("", OpType::Pow, std::vector<TensorPtr>{a},
                                  std::vector<float>{b}, a->rows(), a->cols(),
                                  false);
}

// Print Statements

void Tensor::pShape() const {
  std::cout << m_label << " (" << m_rows << " x " << m_cols << ")";
}

void Tensor::pData() const {
  std::vector<float> data = toHost();

  pShape();
  std::cout << "[\n";
  for (int i = 0; i < m_rows; i++) {
    std::cout << "\t[ ";
    for (int j = 0; j < m_cols; j++) {
      std::cout << data[i * m_cols + j] << " ";
    }
    std::cout << "],\n";
  }
  std::cout << "]" << std::endl;
}
