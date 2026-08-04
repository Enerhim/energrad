#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "tensor.cuh"
#include "utils.cuh"

static constexpr float kEps = 1e-4f;

size_t numel(const Tensor &t) {
  size_t n = 1;
  for (size_t i = 0; i < t->rank; ++i)
    n *= t->shape[i];
  return n;
}

float logical_at(const Tensor &t, size_t flat_idx) {
  size_t offset = 0;
  for (size_t d = 0; d < t->rank; ++d) {
    size_t stride = 1;
    for (size_t k = d + 1; k < t->rank; ++k)
      stride *= t->shape[k];
    size_t coord = flat_idx / stride;
    flat_idx %= stride;
    offset += coord * t->strides[d];
  }
  return t->storage->data[offset];
}

void require(bool cond, const std::string &msg) {
  if (!cond) {
    std::cerr << "[FAIL] " << msg << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void expect_shape(const Tensor &t, const std::vector<size_t> &shape,
                  const std::string &name) {
  require(t != nullptr, name + ": tensor is null");
  require(t->rank == shape.size(), name + ": rank mismatch (got " +
                                       std::to_string(t->rank) + ", expected " +
                                       std::to_string(shape.size()) + ")");
  for (size_t i = 0; i < shape.size(); ++i) {
    require(t->shape[i] == shape[i],
            name + ": shape mismatch at dim " + std::to_string(i) + " (got " +
                std::to_string(t->shape[i]) + ", expected " +
                std::to_string(shape[i]) + ")");
  }
}

void expect_close(float got, float expected, const std::string &name,
                  size_t idx = 0, float eps = kEps) {
  if (std::isnan(expected)) {
    require(std::isnan(got),
            name + ": expected NaN at index " + std::to_string(idx));
    return;
  }
  if (std::fabs(got - expected) > eps) {
    std::cerr << "[FAIL] " << name << " idx=" << idx << " got=" << got
              << " expected=" << expected
              << " diff=" << std::fabs(got - expected) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void expect_tensor_close(const Tensor &t, const std::vector<float> &expected,
                         const std::string &name, float eps = kEps) {
  require(t != nullptr, name + ": tensor is null");
  const size_t n = numel(t);
  require(n == expected.size(), name + ": element count mismatch (got " +
                                    std::to_string(n) + ", expected " +
                                    std::to_string(expected.size()) + ")");
  for (size_t i = 0; i < n; ++i) {
    expect_close(logical_at(t, i), expected[i], name, i, eps);
  }
}

Tensor make_tensor(const std::vector<size_t> &shape,
                   const std::vector<float> &values) {
  auto t = TensorZeros(shape, StorageDevice::CPU);
  require(numel(t) == values.size(),
          "make_tensor: values size does not match shape");
  for (size_t i = 0; i < values.size(); ++i)
    t->storage->data[i] = values[i];
  return t;
}

void test_init_ops() {
  auto t1 =
      TensorCreate(std::vector<size_t>{1, 2, 3}, 1.5f, StorageDevice::CPU);
  auto t2 = TensorOnes(std::vector<size_t>{1, 2, 3}, StorageDevice::CPU);
  auto t3 = TensorZeros(std::vector<size_t>{4, 1, 2, 2}, StorageDevice::CPU);
  auto t4 = TensorCopy(t1);

  expect_shape(t1, {1, 2, 3}, "TensorCreate");
  expect_shape(t2, {1, 2, 3}, "TensorOnes");
  expect_shape(t3, {4, 1, 2, 2}, "TensorZeros");

  expect_tensor_close(t1, {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f}, "TensorCreate");
  expect_tensor_close(t2, {1, 1, 1, 1, 1, 1}, "TensorOnes");
  expect_tensor_close(t3, std::vector<float>(16, 0.0f), "TensorZeros");
  expect_tensor_close(t4, {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f}, "TensorCopy");
}

void test_unary_ops() {
  auto pos = make_tensor({1, 2, 3}, {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
  auto neg = make_tensor({1, 2, 3}, {-1.5f, -1.5f, -1.5f, -1.5f, -1.5f, -1.5f});
  auto mix = make_tensor({1, 2, 3}, {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.0f});
  auto softmax_in = make_tensor({2, 3}, {1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f});

  expect_tensor_close(TensorNeg(pos),
                      {-1.5f, -1.5f, -1.5f, -1.5f, -1.5f, -1.5f}, "TensorNeg");
  expect_tensor_close(TensorAbs(neg), {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f},
                      "TensorAbs");

  const float s = std::sqrt(1.5f);
  expect_tensor_close(TensorSqrt(pos), {s, s, s, s, s, s}, "TensorSqrt");
  expect_tensor_close(TensorReciprocal(pos),
                      {2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f,
                       2.0f / 3.0f, 2.0f / 3.0f},
                      "TensorReciprocal");

  const float e = std::exp(1.5f);
  expect_tensor_close(TensorExp(pos), {e, e, e, e, e, e}, "TensorExp");
  expect_tensor_close(TensorNLog(pos),
                      {std::log(1.5f), std::log(1.5f), std::log(1.5f),
                       std::log(1.5f), std::log(1.5f), std::log(1.5f)},
                      "TensorNLog");
  expect_tensor_close(TensorLog2(pos),
                      {std::log2(1.5f), std::log2(1.5f), std::log2(1.5f),
                       std::log2(1.5f), std::log2(1.5f), std::log2(1.5f)},
                      "TensorLog2");
  expect_tensor_close(TensorLog10(pos),
                      {std::log10(1.5f), std::log10(1.5f), std::log10(1.5f),
                       std::log10(1.5f), std::log10(1.5f), std::log10(1.5f)},
                      "TensorLog10");

  expect_tensor_close(TensorSin(pos),
                      {std::sin(1.5f), std::sin(1.5f), std::sin(1.5f),
                       std::sin(1.5f), std::sin(1.5f), std::sin(1.5f)},
                      "TensorSin");
  expect_tensor_close(TensorSinh(pos),
                      {std::sinh(1.5f), std::sinh(1.5f), std::sinh(1.5f),
                       std::sinh(1.5f), std::sinh(1.5f), std::sinh(1.5f)},
                      "TensorSinh");
  expect_tensor_close(TensorCos(pos),
                      {std::cos(1.5f), std::cos(1.5f), std::cos(1.5f),
                       std::cos(1.5f), std::cos(1.5f), std::cos(1.5f)},
                      "TensorCos");
  expect_tensor_close(TensorCosh(pos),
                      {std::cosh(1.5f), std::cosh(1.5f), std::cosh(1.5f),
                       std::cosh(1.5f), std::cosh(1.5f), std::cosh(1.5f)},
                      "TensorCosh");
  expect_tensor_close(TensorTan(pos),
                      {std::tan(1.5f), std::tan(1.5f), std::tan(1.5f),
                       std::tan(1.5f), std::tan(1.5f), std::tan(1.5f)},
                      "TensorTan");
  expect_tensor_close(TensorTanh(pos),
                      {std::tanh(1.5f), std::tanh(1.5f), std::tanh(1.5f),
                       std::tanh(1.5f), std::tanh(1.5f), std::tanh(1.5f)},
                      "TensorTanh");

  expect_tensor_close(TensorFloor(mix), {-2, -1, 0, 0, 1, 2}, "TensorFloor");
  expect_tensor_close(TensorCeil(mix), {-1, 0, 0, 1, 2, 2}, "TensorCeil");
  expect_tensor_close(TensorScale(pos, 3.0f),
                      {4.5f, 4.5f, 4.5f, 4.5f, 4.5f, 4.5f}, "TensorScale");
  expect_tensor_close(TensorRelu(mix), {0, 0, 0, 0.5f, 1.5f, 2.0f},
                      "TensorRelu");
  expect_tensor_close(TensorSigmoid(pos),
                      {0.81757444f, 0.81757444f, 0.81757444f, 0.81757444f,
                       0.81757444f, 0.81757444f},
                      "TensorSigmoid", 1e-4f);

  auto sm = TensorSoftmax(softmax_in);
  expect_shape(sm, {2, 3}, "TensorSoftmax");
  expect_tensor_close(sm,
                      {0.09003057f, 0.24472848f, 0.66524094f, 0.66524094f,
                       0.24472848f, 0.09003057f},
                      "TensorSoftmax", 1e-4f);
}

void test_binary_ops() {
  auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = make_tensor({2, 3}, {6, 5, 4, 3, 2, 1});

  expect_tensor_close(TensorAdd(a, b), {7, 7, 7, 7, 7, 7}, "TensorAdd");
  expect_tensor_close(TensorSub(a, b), {-5, -3, -1, 1, 3, 5}, "TensorSub");
  expect_tensor_close(TensorMul(a, b), {6, 10, 12, 12, 10, 6}, "TensorMul");

  auto m1 = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto m2 = make_tensor({3, 2}, {7, 8, 9, 10, 11, 12});
  auto mm = TensorMatmul(m1, m2);
  expect_shape(mm, {2, 2}, "TensorMatmul");
  expect_tensor_close(mm, {58, 64, 139, 154}, "TensorMatmul");
}

void test_shape_ops() {
  auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto t = TensorTranspose(a, 0, 1);
  expect_shape(t, {3, 2}, "TensorTranspose");
  expect_tensor_close(t, {1, 4, 2, 5, 3, 6}, "TensorTranspose");

  std::array<size_t, MAX_DIMS> target{};
  target.fill(1);
  target[0] = 2;
  target[1] = 3;
  auto base = make_tensor({1, 3}, {7, 8, 9});
  auto b = TensorBroadcast(base, target);
  expect_shape(b, {2, 3}, "TensorBroadcast");
  expect_tensor_close(b, {7, 8, 9, 7, 8, 9}, "TensorBroadcast");

  auto c = TensorContiguous(t);
  expect_shape(c, {3, 2}, "TensorContiguous");
  expect_tensor_close(c, {1, 4, 2, 5, 3, 6}, "TensorContiguous");
}

int main() {
  test_init_ops();
  test_unary_ops();
  test_binary_ops();
  test_shape_ops();

  std::cout << "ALL TESTS PASSED\n";
  return 0;
}
