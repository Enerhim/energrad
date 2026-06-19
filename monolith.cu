#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

enum class MemoryKind { Host, Device };

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

// Operation Class (Nodes in DAG)
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

// Cuda Kernels

__global__ void tensor_add_kernel(const float *A, const float *B, float *C,
                                  size_t N) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void tensor_accumulate_kernel(const float *A, float *B, size_t N) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    B[idx] += A[idx];
  }
}

__global__ void tensor_set(float fill, float *A, size_t N) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    A[idx] = fill;
  }
}

// Memory Pooling in CUDA

struct CudaContext {
  int device = 0;
  cudaStream_t stream{};

  explicit CudaContext(int device = 0) : device(device) {
    cudaSetDevice(device);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, device);

    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
  }

  ~CudaContext() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }
};

// Tensor Object

class TensorObject : public std::enable_shared_from_this<TensorObject> {

public:
  TensorObject(const std::string &label, const std::vector<size_t> &shape,
               bool hasGrad, const std::vector<float> &data,
               std::shared_ptr<CudaContext> ctx)
      : label(label), shape(shape), hasGrad(hasGrad), ctx(ctx) {

    // Construct Strides
    strides.resize(shape.size());
    size_t stride = 1;
    for (size_t i = shape.size() - 1; i > 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    strides[0] = stride;
    _elements = strides[0] * shape[0];
    _size = _elements * sizeof(float);

    // Allocate memory on GPU
    cudaMallocAsync(&data_buffer.ptr, _size, ctx->stream);

    if (!data.empty()) {
      if (data.size() != _elements) {
        // std::cout << "_elements: " << _elements
        //           << " | data.size(): " << data.size() << std::endl;
        throw std::runtime_error("Error: Data provided as argument to Tensor "
                                 "is neither empty nor correct size.\n");
      }
      cudaMemcpyAsync(data_buffer.ptr, data.data(), _size,
                      cudaMemcpyHostToDevice, ctx->stream);
    }
  }

  ~TensorObject() {
    if (data_buffer.kind == MemoryKind::Device)
      cudaFreeAsync(data_buffer.ptr, ctx->stream);
    if (grad_buffer.kind == MemoryKind::Device)
      cudaFreeAsync(grad_buffer.ptr, ctx->stream);
  }

  std::shared_ptr<CudaContext> getCudaContext() const { return ctx; }

  Buffer deviceBuffer() const { return data_buffer; }
  Buffer deviceGradBuffer() const { return grad_buffer; }

  std::vector<float> hostBuffer() {
    std::vector<float> h(_elements);
    cudaMemcpyAsync(h.data(), data_buffer.ptr, _size, cudaMemcpyDeviceToHost,
                    ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return h;
  }

  std::vector<float> hostGradBuffer() {
    if (grad_buffer.ptr == nullptr || !hasGrad)
      return {};

    std::vector<float> h(_elements);
    cudaMemcpyAsync(h.data(), grad_buffer.ptr, _size, cudaMemcpyDeviceToHost,
                    ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return h;
  }

  const std::vector<size_t> &getShape() const { return shape; }
  size_t getSize() const { return _size; }
  size_t noElements() const { return _elements; }
  bool hasGradient() const { return hasGrad; }
  void setLabel(const std::string &l) { label = l; }
  const std::string &getLabel() const { return label; }

  friend Tensor operator+(const Tensor &, const Tensor &);

  void setGrad(const Buffer &buffer) {

    if (hasGrad) {

      if (grad_buffer.ptr == nullptr)
        cudaMallocAsync(&grad_buffer.ptr, _size, ctx->stream);
      switch (buffer.kind) {
      case MemoryKind::Host:
        cudaMemcpyAsync(grad_buffer.ptr, buffer.ptr, _size,
                        cudaMemcpyHostToDevice, ctx->stream);
        break;
      case MemoryKind::Device:
        cudaMemcpyAsync(grad_buffer.ptr, buffer.ptr, _size,
                        cudaMemcpyDeviceToDevice, ctx->stream);
        break;
      }
    } else {
      throw std::runtime_error("hasGrad = false for this Tensor!\n");
    }
  }

  void zeroGrad() {

    if (hasGrad) {

      if (grad_buffer.ptr == nullptr)
        cudaMallocAsync(&grad_buffer.ptr, _size, ctx->stream);

      cudaMemsetAsync(grad_buffer.ptr, 0.0f, _size, ctx->stream);
    } else {
      throw std::runtime_error("hasGrad = false for this Tensor!\n");
    }
  }

  void accumulateGrad(const Buffer &top_gradient) {
    if (!hasGrad)
      return;

    if (grad_buffer.ptr == nullptr) {
      cudaMallocAsync(&grad_buffer.ptr, _size, ctx->stream);
      cudaMemsetAsync(grad_buffer.ptr, 0, _size, ctx->stream);
      grad_buffer.kind = MemoryKind::Device;
    }

    uint N = _elements;
    const uint BLOCK_SIZE = 32;
    uint blocks = CEIL_DIV(N, BLOCK_SIZE);

    tensor_accumulate_kernel<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
        top_gradient.ptr, grad_buffer.ptr, N);
  }

  std::shared_ptr<Operation> getOperation() const { return parent_op; }

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

  void setOperation(std::shared_ptr<Operation> op) { parent_op = op; }
};

// Operation Sub Classes

class AddOp : public Operation {
public:
  void backward(const Buffer &top_gradient) override {
    if (top_gradient.kind != MemoryKind::Device)
      throw std::runtime_error("Buffer kind for AddOp is not device.\n");

    auto a = parents[0].lock();
    auto b = parents[1].lock();

    a->accumulateGrad(top_gradient);
    b->accumulateGrad(top_gradient);
  }
};

Tensor operator+(const Tensor &a, const Tensor &b) {
  if (a->getShape() != b->getShape()) {
    throw std::runtime_error(
        "Error: Shapes do not match during + operation.\n");
  }

  auto ctx = a->getCudaContext();
  Tensor result = std::make_shared<TensorObject>(
      "", a->getShape(), a->hasGradient() || b->hasGradient(),
      std::vector<float>{}, ctx);

  auto op = std::make_shared<AddOp>();
  op->setParents({a, b});
  result->setOperation(op);

  uint N = a->noElements();
  const uint BLOCK_SIZE = 32;
  uint blocks = CEIL_DIV(N, BLOCK_SIZE);

  tensor_add_kernel<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
      a->deviceBuffer().ptr, b->deviceBuffer().ptr, result->deviceBuffer().ptr,
      N);
  return result;
}

Tensor make_tensor(const std::string &label, const std::vector<size_t> &shape,
                   bool hasGrad, const std::vector<float> &data,
                   std::shared_ptr<CudaContext> ctx) {
  return std::make_shared<TensorObject>(label, shape, hasGrad, data, ctx);
}

class Engine {
public:
  void backward(const Tensor &root) {
    indeg_count.clear();
    visited.clear();

    indeg_count[root.get()] = 0;
    count(root.get());

    size_t N = root->noElements();
    const uint BLOCK_SIZE = 32;
    uint blocks = CEIL_DIV(N, BLOCK_SIZE);
    root->zeroGrad();
    tensor_set<<<blocks, BLOCK_SIZE, 0, root->getCudaContext()->stream>>>(
        1.0f, root->deviceGradBuffer().ptr, N);

    std::queue<TensorObject *> q;

    for (auto it = indeg_count.begin(); it != indeg_count.end(); it++) {
      if (indeg_count[it->first] == 0)
        q.push(it->first);
    }
    std::vector<TensorObject *> topo;

    while (!q.empty()) {
      auto top = q.front();
      q.pop();

      topo.push_back(top);
      auto op = top->getOperation();
      if (!op)
        continue;

      auto parents = op->getParents();
      for (auto &wp : parents) {
        if (auto p = wp.lock()) {
          auto t = p.get();
          indeg_count[t]--;
          if (indeg_count[t] == 0) {
            q.push(t);
          }
        }
      }
    }

    for (auto t : topo) {
      auto op = t->getOperation();
      if (!op)
        continue;

      auto grad = t->deviceGradBuffer();
      if (grad.ptr == nullptr)
        continue;

      op->backward(grad);
    }
  }

private:
  std::map<TensorObject *, uint> indeg_count;
  std::unordered_set<TensorObject *> visited;

  void count(TensorObject *root) {
    if (visited.find(root) != visited.end())
      return;
    visited.insert(root);

    auto op = root->getOperation();
    if (!op)
      return;

    auto parents = op->getParents();
    for (auto &wp : parents) {
      if (auto p = wp.lock()) {
        auto t = p.get();
        indeg_count[t]++;
        count(t);
      }
    }
  }
};

int main() {
  std::shared_ptr<CudaContext> ctx = std::make_shared<CudaContext>();
  Engine engine;

  Tensor A = make_tensor("A", {2, 2}, true, {1, 2, 3, 4}, ctx);
  Tensor B = make_tensor("B", {2, 2}, true, {1, 2, 3, 4}, ctx);

  Tensor C = A + B;
  cudaStreamSynchronize(ctx->stream);

  engine.backward(C);
  engine.backward(C);

  for (float x : A->hostGradBuffer()) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  for (float x : B->hostGradBuffer()) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  for (float x : C->hostGradBuffer()) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  // for (float x : D->hostGradBuffer()) {
  //   std::cout << x << " ";
  // }
  // std::cout << std::endl;
  return 0;
}
