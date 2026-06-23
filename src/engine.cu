#include "../include/engine.cuh"
#include "../include/op.cuh"
#include <queue>

__global__ void tensor_set(float fill, float *A, size_t N) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    A[idx] = fill;
  }
}

void Engine::backward(const Tensor &root) {
  indeg_count.clear();
  visited.clear();

  indeg_count[root.get()] = 0;
  count(root.get());

  size_t N = root->noElements();
  const uint BLOCK_SIZE = 32;
  uint blocks = CEIL_DIV(N, BLOCK_SIZE);
  root->zeroGrad();
  tensor_set<<<blocks, BLOCK_SIZE, 0, root->getCudaContext()->stream>>>(
      1.0f, root->deviceGrad().data(), N);

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

    auto grad = t->deviceGrad();
    if (grad.data() == nullptr)
      continue;

    op->backward(grad);
  }
}

void Engine::count(TensorObject *root) {
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
