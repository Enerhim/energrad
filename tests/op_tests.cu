#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "tensor.cuh"
#include "utils.cuh"

static constexpr float kEps = 1e-4f;

static int g_failures = 0;
static std::string g_current_test;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

size_t numel(const Tensor &t) {
  size_t n = 1;
  for (size_t i = 0; i < t->rank; ++i)
    n *= t->shape[i];
  return n;
}

// Reads element `flat_idx` in logical (row-major) order, honouring strides so
// that views (transpose / broadcast) are checked through their own layout.
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

void fail(const std::string &msg) {
  std::cerr << "[FAIL] " << g_current_test << ": " << msg << '\n';
  ++g_failures;
}

void require(bool cond, const std::string &msg) {
  if (!cond)
    fail(msg);
}

void expect_shape(const Tensor &t, const std::vector<size_t> &shape,
                  const std::string &name) {
  if (t == nullptr) {
    fail(name + ": tensor is null");
    return;
  }
  if (t->rank != shape.size()) {
    fail(name + ": rank mismatch (got " + std::to_string(t->rank) +
         ", expected " + std::to_string(shape.size()) + ")");
    return;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (t->shape[i] != shape[i])
      fail(name + ": shape mismatch at dim " + std::to_string(i) + " (got " +
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
    std::cerr << "[FAIL] " << g_current_test << ": " << name
              << " idx=" << idx << " got=" << got << " expected=" << expected
              << " diff=" << std::fabs(got - expected) << '\n';
    ++g_failures;
  }
}

void expect_tensor_close(const Tensor &t, const std::vector<float> &expected,
                         const std::string &name, float eps = kEps) {
  if (t == nullptr) {
    fail(name + ": tensor is null");
    return;
  }
  const size_t n = numel(t);
  if (n != expected.size()) {
    fail(name + ": element count mismatch (got " + std::to_string(n) +
         ", expected " + std::to_string(expected.size()) + ")");
    return;
  }
  for (size_t i = 0; i < n; ++i)
    expect_close(logical_at(t, i), expected[i], name, i, eps);
}

// Builds a contiguous CPU tensor through the public API.
Tensor make_tensor(const std::vector<size_t> &shape,
                   const std::vector<float> &values) {
  auto t = TensorZeros(shape, StorageDevice::CPU);
  require(numel(t) == values.size(),
          "make_tensor: values size does not match shape");
  TensorLoadData(t, values.data(), values.size());
  return t;
}

std::array<size_t, MAX_DIMS> shape_array(const std::vector<size_t> &shape) {
  std::array<size_t, MAX_DIMS> arr{};
  arr.fill(0);
  std::copy(shape.begin(), shape.end(), arr.begin());
  return arr;
}

// Applies `f` elementwise to `in` and compares against the op output.
void expect_unary(Tensor out, const std::vector<float> &in,
                  float (*f)(float), const std::string &name,
                  float eps = kEps) {
  std::vector<float> expected(in.size());
  for (size_t i = 0; i < in.size(); ++i)
    expected[i] = f(in[i]);
  expect_tensor_close(out, expected, name, eps);
}

// ---------------------------------------------------------------------------
// IO / init ops
// ---------------------------------------------------------------------------

void test_init_ops() {
  g_current_test = "init_ops";

  auto t1 = TensorCreate(std::vector<size_t>{1, 2, 3}, 1.5f, StorageDevice::CPU);
  auto t2 = TensorOnes(std::vector<size_t>{1, 2, 3}, StorageDevice::CPU);
  auto t3 = TensorZeros(std::vector<size_t>{4, 1, 2, 2}, StorageDevice::CPU);

  expect_shape(t1, {1, 2, 3}, "TensorCreate");
  expect_shape(t2, {1, 2, 3}, "TensorOnes");
  expect_shape(t3, {4, 1, 2, 2}, "TensorZeros");

  expect_tensor_close(t1, std::vector<float>(6, 1.5f), "TensorCreate");
  expect_tensor_close(t2, std::vector<float>(6, 1.0f), "TensorOnes");
  expect_tensor_close(t3, std::vector<float>(16, 0.0f), "TensorZeros");

  // Default strides are row-major contiguous.
  require(checkContiguous(t3), "TensorCreate: fresh tensor is not contiguous");
  require(t3->strides[0] == 4 && t3->strides[1] == 4 && t3->strides[2] == 2 &&
              t3->strides[3] == 1,
          "TensorCreate: unexpected strides");
  require(t3->storage->bytes == 16 * sizeof(float),
          "TensorCreate: unexpected storage size");

  // TensorLoadData copies element-for-element.
  auto loaded = TensorZeros(std::vector<size_t>{2, 3}, StorageDevice::CPU);
  const std::vector<float> data{1, 2, 3, 4, 5, 6};
  TensorLoadData(loaded, data.data(), data.size());
  expect_tensor_close(loaded, data, "TensorLoadData");

  // Size mismatch is a no-op, not a buffer overrun.
  const std::vector<float> tooShort{9, 9};
  TensorLoadData(loaded, tooShort.data(), tooShort.size());
  expect_tensor_close(loaded, data, "TensorLoadData(size mismatch)");

  // TensorCopy default: shares storage (a view).
  auto view = TensorCopy(loaded);
  expect_shape(view, {2, 3}, "TensorCopy(view)");
  require(view->storage == loaded->storage,
          "TensorCopy(view): storage is not shared");
  expect_tensor_close(view, data, "TensorCopy(view)");

  // TensorCopy(newStorage): deep copy, mutation does not alias back.
  auto deep = TensorCopy(loaded, true);
  expect_shape(deep, {2, 3}, "TensorCopy(deep)");
  require(deep->storage != loaded->storage,
          "TensorCopy(deep): storage was shared");
  expect_tensor_close(deep, data, "TensorCopy(deep)");
  deep->storage->data[0] = -42.0f;
  expect_close(logical_at(loaded, 0), 1.0f, "TensorCopy(deep) aliasing", 0);

  // TensorRand: shape, and uniform values inside [min, max].
  auto r = TensorRand(std::vector<size_t>{2, 3, 4}, StorageDevice::CPU, false,
                      -2.0f, 5.0f);
  expect_shape(r, {2, 3, 4}, "TensorRand(uniform)");
  bool inRange = true, allSame = true;
  const float first = logical_at(r, 0);
  for (size_t i = 0; i < numel(r); ++i) {
    const float v = logical_at(r, i);
    if (v < -2.0f || v > 5.0f)
      inRange = false;
    if (v != first)
      allSame = false;
  }
  require(inRange, "TensorRand(uniform): value outside [min, max]");
  require(!allSame, "TensorRand(uniform): all values identical");

  auto rn = TensorRand(std::vector<size_t>{4, 4}, StorageDevice::CPU, true,
                       0.0f, 1.0f);
  expect_shape(rn, {4, 4}, "TensorRand(normal)");
  bool finite = true;
  for (size_t i = 0; i < numel(rn); ++i)
    if (!std::isfinite(logical_at(rn, i)))
      finite = false;
  require(finite, "TensorRand(normal): non-finite sample");
}

// ---------------------------------------------------------------------------
// Unary ops
// ---------------------------------------------------------------------------

void test_unary_ops() {
  g_current_test = "unary_ops";

  const std::vector<float> posv{0.5f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f};
  const std::vector<float> mixv{-1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.0f};

  auto pos = make_tensor({1, 2, 3}, posv);
  auto mix = make_tensor({1, 2, 3}, mixv);
  auto neg = make_tensor({1, 2, 3}, {-1.5f, -1.5f, -1.5f, -1.5f, -1.5f, -1.5f});

  expect_unary(TensorNeg(mix), mixv, [](float x) { return -x; }, "TensorNeg");
  expect_tensor_close(TensorAbs(neg), std::vector<float>(6, 1.5f), "TensorAbs");
  expect_unary(TensorAbs(mix), mixv, [](float x) { return std::fabs(x); },
               "TensorAbs(mix)");

  expect_unary(TensorSqrt(pos), posv, [](float x) { return std::sqrt(x); },
               "TensorSqrt");
  expect_unary(TensorReciprocal(pos), posv, [](float x) { return 1.0f / x; },
               "TensorReciprocal");
  expect_unary(TensorExp(pos), posv, [](float x) { return std::exp(x); },
               "TensorExp");
  expect_unary(TensorNLog(pos), posv, [](float x) { return std::log(x); },
               "TensorNLog");
  expect_unary(TensorLog2(pos), posv, [](float x) { return std::log2(x); },
               "TensorLog2");
  expect_unary(TensorLog10(pos), posv, [](float x) { return std::log10(x); },
               "TensorLog10");

  expect_unary(TensorSin(pos), posv, [](float x) { return std::sin(x); },
               "TensorSin");
  expect_unary(TensorSinh(pos), posv, [](float x) { return std::sinh(x); },
               "TensorSinh");
  expect_unary(TensorCos(pos), posv, [](float x) { return std::cos(x); },
               "TensorCos");
  expect_unary(TensorCosh(pos), posv, [](float x) { return std::cosh(x); },
               "TensorCosh");
  expect_unary(TensorTan(pos), posv, [](float x) { return std::tan(x); },
               "TensorTan");
  expect_unary(TensorTanh(pos), posv, [](float x) { return std::tanh(x); },
               "TensorTanh");

  expect_tensor_close(TensorFloor(mix), {-2, -1, 0, 0, 1, 2}, "TensorFloor");
  expect_tensor_close(TensorCeil(mix), {-1, 0, 0, 1, 2, 2}, "TensorCeil");
  expect_unary(TensorScale(mix, 3.0f), mixv, [](float x) { return 3.0f * x; },
               "TensorScale");
  expect_unary(TensorScale(mix, 0.0f), mixv, [](float) { return 0.0f; },
               "TensorScale(zero)");
  expect_tensor_close(TensorRelu(mix), {0, 0, 0, 0.5f, 1.5f, 2.0f},
                      "TensorRelu");
  expect_unary(TensorSigmoid(mix), mixv,
               [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
               "TensorSigmoid");

  // Result of a unary op is a fresh, contiguous tensor with its own storage.
  auto negated = TensorNeg(pos);
  require(negated->storage != pos->storage,
          "TensorNeg: result aliases the input storage");
  require(checkContiguous(negated), "TensorNeg: result is not contiguous");
  expect_shape(negated, {1, 2, 3}, "TensorNeg");

  // Unary ops must honour strides: run one on a transposed (non-contiguous)
  // view.
  auto base = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto tView = TensorTranspose(base, 0, 1);
  expect_tensor_close(TensorNeg(tView), {-1, -4, -2, -5, -3, -6},
                      "TensorNeg(transposed)");
  expect_tensor_close(TensorScale(tView, 2.0f), {2, 8, 4, 10, 6, 12},
                      "TensorScale(transposed)");

  // Softmax: rows of the last two dims, batched.
  auto softmaxIn = make_tensor({2, 3}, {1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f});
  auto sm = TensorSoftmax(softmaxIn);
  expect_shape(sm, {2, 3}, "TensorSoftmax");
  expect_tensor_close(sm,
                      {0.09003057f, 0.24472848f, 0.66524094f, 0.66524094f,
                       0.24472848f, 0.09003057f},
                      "TensorSoftmax");

  // Softmax rows sum to 1 and are shift invariant.
  auto shifted = TensorSoftmax(TensorAdd(
      softmaxIn, TensorCreate(std::vector<size_t>{2, 3}, 100.0f,
                              StorageDevice::CPU)));
  expect_tensor_close(shifted,
                      {0.09003057f, 0.24472848f, 0.66524094f, 0.66524094f,
                       0.24472848f, 0.09003057f},
                      "TensorSoftmax(shift invariance)");

  auto smBatched = TensorSoftmax(make_tensor(
      {2, 1, 2}, {0.0f, std::log(3.0f), std::log(3.0f), 0.0f}));
  expect_shape(smBatched, {2, 1, 2}, "TensorSoftmax(batched)");
  expect_tensor_close(smBatched, {0.25f, 0.75f, 0.75f, 0.25f},
                      "TensorSoftmax(batched)");

  // Backward helpers (elementwise local derivatives).
  expect_tensor_close(TensorAbsBackward(mix), {-1, -1, 0, 1, 1, 1},
                      "TensorAbsBackward");
  expect_tensor_close(TensorReluBackward(mix), {0, 0, 0, 1, 1, 1},
                      "TensorReluBackward");

  // Softmax backward: s * (1 - s) on the already-softmaxed values.
  auto smVals = make_tensor({1, 3}, {0.1f, 0.3f, 0.6f});
  expect_tensor_close(TensorSoftmaxBackward(smVals),
                      {0.1f * 0.9f, 0.3f * 0.7f, 0.6f * 0.4f},
                      "TensorSoftmaxBackward");
}

// ---------------------------------------------------------------------------
// Binary ops
// ---------------------------------------------------------------------------

void test_binary_ops() {
  g_current_test = "binary_ops";

  auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = make_tensor({2, 3}, {6, 5, 4, 3, 2, 1});

  expect_shape(TensorAdd(a, b), {2, 3}, "TensorAdd");
  expect_tensor_close(TensorAdd(a, b), {7, 7, 7, 7, 7, 7}, "TensorAdd");
  expect_tensor_close(TensorSub(a, b), {-5, -3, -1, 1, 3, 5}, "TensorSub");
  expect_tensor_close(TensorMul(a, b), {6, 10, 12, 12, 10, 6}, "TensorMul");

  // Binary result owns fresh storage.
  auto sum = TensorAdd(a, b);
  require(sum->storage != a->storage && sum->storage != b->storage,
          "TensorAdd: result aliases an input storage");
  require(checkContiguous(sum), "TensorAdd: result is not contiguous");

  // Implicit broadcasting of B against A (same rank, size-1 dims).
  auto row = make_tensor({1, 3}, {10, 20, 30});
  expect_tensor_close(TensorAdd(a, row), {11, 22, 33, 14, 25, 36},
                      "TensorAdd(broadcast row)");
  expect_tensor_close(TensorSub(a, row), {-9, -18, -27, -6, -15, -24},
                      "TensorSub(broadcast row)");
  expect_tensor_close(TensorMul(a, row), {10, 40, 90, 40, 100, 180},
                      "TensorMul(broadcast row)");

  auto col = make_tensor({2, 1}, {10, 100});
  expect_tensor_close(TensorAdd(a, col), {11, 12, 13, 104, 105, 106},
                      "TensorAdd(broadcast col)");

  // Broadcasting a scalar-shaped tensor.
  auto scalarLike = make_tensor({1, 1}, {7});
  expect_tensor_close(TensorMul(a, scalarLike), {7, 14, 21, 28, 35, 42},
                      "TensorMul(broadcast scalar)");

  // Elementwise ops honour strides on the left operand too.
  auto aT = TensorTranspose(a, 0, 1);
  auto bT = TensorTranspose(b, 0, 1);
  expect_shape(TensorAdd(aT, bT), {3, 2}, "TensorAdd(transposed)");
  expect_tensor_close(TensorAdd(aT, bT), {7, 7, 7, 7, 7, 7},
                      "TensorAdd(transposed)");

  // Matmul: (2x3) * (3x2) = (2x2)
  auto m1 = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto m2 = make_tensor({3, 2}, {7, 8, 9, 10, 11, 12});
  auto mm = TensorMatmul(m1, m2);
  expect_shape(mm, {2, 2}, "TensorMatmul");
  expect_tensor_close(mm, {58, 64, 139, 154}, "TensorMatmul");

  // Identity matmul leaves the operand unchanged.
  auto ident = make_tensor({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
  expect_tensor_close(TensorMatmul(m1, ident), {1, 2, 3, 4, 5, 6},
                      "TensorMatmul(identity)");

  // Non-square inner dim: (1x4) * (4x3) = (1x3)
  auto v = make_tensor({1, 4}, {1, 2, 3, 4});
  auto w = make_tensor({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto vw = TensorMatmul(v, w);
  expect_shape(vw, {1, 3}, "TensorMatmul(1x4 * 4x3)");
  expect_tensor_close(vw, {70, 80, 90}, "TensorMatmul(1x4 * 4x3)");

  // Batched matmul: (2x2x3) * (2x3x2) = (2x2x2)
  auto ba = make_tensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 0, 0, 1, 0});
  auto bb = make_tensor({2, 3, 2}, {7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6});
  auto bmm = TensorMatmul(ba, bb);
  expect_shape(bmm, {2, 2, 2}, "TensorMatmul(batched)");
  expect_tensor_close(bmm, {58, 64, 139, 154, 1, 2, 3, 4},
                      "TensorMatmul(batched)");

  // Rank < 2 operands are rejected.
  auto rank1 = make_tensor({3}, {1, 2, 3});
  require(TensorMatmul(rank1, m2) == nullptr,
          "TensorMatmul: rank-1 operand should return nullptr");
}

// ---------------------------------------------------------------------------
// Shape ops
// ---------------------------------------------------------------------------

void test_shape_ops() {
  g_current_test = "shape_ops";

  auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});

  // Transpose is a view: shape/strides swap, storage is shared.
  auto t = TensorTranspose(a, 0, 1);
  expect_shape(t, {3, 2}, "TensorTranspose");
  expect_tensor_close(t, {1, 4, 2, 5, 3, 6}, "TensorTranspose");
  require(t->storage == a->storage, "TensorTranspose: storage is not shared");
  require(!checkContiguous(t), "TensorTranspose: view reports contiguous");
  require(t->strides[0] == 1 && t->strides[1] == 3,
          "TensorTranspose: unexpected strides");

  // Double transpose is the identity.
  auto tt = TensorTranspose(t, 0, 1);
  expect_shape(tt, {2, 3}, "TensorTranspose(twice)");
  expect_tensor_close(tt, {1, 2, 3, 4, 5, 6}, "TensorTranspose(twice)");

  // 3D transpose of the two trailing dims.
  auto a3 = make_tensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto t3 = TensorTranspose(a3, 1, 2);
  expect_shape(t3, {2, 3, 2}, "TensorTranspose(3D)");
  expect_tensor_close(t3, {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12},
                      "TensorTranspose(3D)");

  // Broadcast: size-1 dims get a zero stride, storage stays shared.
  auto row = make_tensor({1, 3}, {7, 8, 9});
  auto br = TensorBroadcast(row, shape_array({2, 3}), 2);
  expect_shape(br, {2, 3}, "TensorBroadcast(row)");
  expect_tensor_close(br, {7, 8, 9, 7, 8, 9}, "TensorBroadcast(row)");
  require(br->storage == row->storage,
          "TensorBroadcast: storage is not shared");
  require(br->strides[0] == 0, "TensorBroadcast: broadcast dim stride != 0");

  auto col = make_tensor({2, 1}, {7, 8});
  auto bc = TensorBroadcast(col, shape_array({2, 3}), 2);
  expect_shape(bc, {2, 3}, "TensorBroadcast(col)");
  expect_tensor_close(bc, {7, 7, 7, 8, 8, 8}, "TensorBroadcast(col)");

  auto cell = make_tensor({1, 1, 1}, {5});
  auto bcell = TensorBroadcast(cell, shape_array({2, 2, 2}), 3);
  expect_shape(bcell, {2, 2, 2}, "TensorBroadcast(3D)");
  expect_tensor_close(bcell, std::vector<float>(8, 5.0f),
                      "TensorBroadcast(3D)");

  // checkBroadcastable agrees with what TensorBroadcast accepts.
  require(checkBroadcastable(row, a), "checkBroadcastable: (1,3) -> (2,3)");
  require(!checkBroadcastable(a, row), "checkBroadcastable: (2,3) -> (1,3)");

  // Contiguous materialises a strided view into fresh row-major storage.
  auto c = TensorContiguous(t);
  expect_shape(c, {3, 2}, "TensorContiguous");
  expect_tensor_close(c, {1, 4, 2, 5, 3, 6}, "TensorContiguous");
  require(checkContiguous(c), "TensorContiguous: result is not contiguous");
  require(c->storage != t->storage,
          "TensorContiguous: result aliases the view storage");

  auto cb = TensorContiguous(br);
  expect_shape(cb, {2, 3}, "TensorContiguous(broadcast)");
  expect_tensor_close(cb, {7, 8, 9, 7, 8, 9}, "TensorContiguous(broadcast)");
  require(checkContiguous(cb),
          "TensorContiguous(broadcast): result is not contiguous");
}

// ---------------------------------------------------------------------------
// Reduction ops
// ---------------------------------------------------------------------------

void test_reduction_ops() {
  g_current_test = "reduction_ops";

  auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});

  // Sum over rows: shape (2,3) -> (1,3)
  auto s0 = TensorSum(a, 0, true);
  expect_shape(s0, {1, 3}, "TensorSum(dim=0)");
  expect_tensor_close(s0, {5, 7, 9}, "TensorSum(dim=0)");

  // Sum over columns: shape (2,3) -> (2,1)
  auto s1 = TensorSum(a, 1, true);
  expect_shape(s1, {2, 1}, "TensorSum(dim=1)");
  expect_tensor_close(s1, {6, 15}, "TensorSum(dim=1)");

  // 3D reductions over each axis.
  auto a3 = make_tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});

  auto r0 = TensorSum(a3, 0, true);
  expect_shape(r0, {1, 2, 2}, "TensorSum(3D, dim=0)");
  expect_tensor_close(r0, {6, 8, 10, 12}, "TensorSum(3D, dim=0)");

  auto r1 = TensorSum(a3, 1, true);
  expect_shape(r1, {2, 1, 2}, "TensorSum(3D, dim=1)");
  expect_tensor_close(r1, {4, 6, 12, 14}, "TensorSum(3D, dim=1)");

  auto r2 = TensorSum(a3, 2, true);
  expect_shape(r2, {2, 2, 1}, "TensorSum(3D, dim=2)");
  expect_tensor_close(r2, {3, 7, 11, 15}, "TensorSum(3D, dim=2)");

  // Reducing a size-1 dim is a copy.
  auto row = make_tensor({1, 3}, {1, 2, 3});
  auto rsum = TensorSum(row, 0, true);
  expect_shape(rsum, {1, 3}, "TensorSum(size-1 dim)");
  expect_tensor_close(rsum, {1, 2, 3}, "TensorSum(size-1 dim)");

  // Result owns fresh contiguous storage.
  require(s0->storage != a->storage, "TensorSum: result aliases input storage");
  require(checkContiguous(s0), "TensorSum: result is not contiguous");
}

int main() {
  test_init_ops();
  test_unary_ops();
  test_binary_ops();
  test_shape_ops();
  test_reduction_ops();

  if (g_failures > 0) {
    std::cerr << g_failures << " CHECK(S) FAILED\n";
    return EXIT_FAILURE;
  }

  std::cout << "ALL TESTS PASSED\n";
  return EXIT_SUCCESS;
}
