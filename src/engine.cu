#include "engine.cuh"
#include "op.cuh"
#include "tensor.cuh"
#include <queue>
#include <stdexcept>

void Engine::count(TensorObject *root) {

  if (visited.find(root) != visited.end())
    return;

  visited.insert(root);

  auto grad_ctx = root->getGradContext();
  auto op = grad_ctx->op;
  if (!op)
    return;

  for (auto &p : op->getParents()) {
    auto t = p.get();
    indeg_count[t]++;
    count(t);
  }
}

void Engine::backward(TensorObject *root) {
  // Clear bs
  indeg_count.clear();
  visited.clear();
  indeg_count[root] = 0;

  // Count indegrees
  count(root);

  // Set root gradient = 1s;
  auto root_grad_ctx = root->getGradContext();
  if (!root_grad_ctx)
    throw std::runtime_error(
        "Error: No Grad Context on root tensor during Engine.backward().\n");
  if (!root_grad_ctx->hasGrad)
    throw std::runtime_error(
        "Error: `hasGrad` = false for root tensor during Engine.backward().\n");

  auto root_storage = root->getStorage();
  auto root_cuda_ctx = root_storage->getCudaContext();
  auto root_shape = root->getShape();

  auto root_grad_tensor = root_grad_ctx->grad;
  if (!root_grad_tensor) {
    allocateGrad(root_shape, 1.0f, root_grad_ctx, root_cuda_ctx);
    root_grad_tensor = root_grad_ctx->grad;
  } else {
    auto root_grad_shape = root_grad_tensor->getShape();
    size_t no_elements = 1;
    for (size_t s : root_grad_shape)
      no_elements *= s;

    auto root_grad_storage = root_grad_tensor->getStorage();
    root_grad_storage->setData(std::vector<float>(no_elements, 1.0f));
  }

  std::queue<TensorObject *> q;

  // Building Topological Order
  for (auto it = indeg_count.begin(); it != indeg_count.end(); it++) {
    if (indeg_count[it->first] == 0)
      q.push(it->first);
  }

  std::vector<TensorObject *> topo;

  while (!q.empty()) {
    auto top = q.front();
    q.pop();

    topo.push_back(top);
    auto top_grad_ctx = top->getGradContext();
    auto op = top_grad_ctx->op;
    if (!op)
      continue;

    auto parents = op->getParents();
    for (auto &p : parents) {
      auto t = p.get();
      indeg_count[t]--;
      if (indeg_count[t] == 0) {
        q.push(t);
      }
    }
  }

  // Backward && Accumulate
  for (auto t : topo) {

    // get Op of current Tensor
    auto grad_ctx = t->getGradContext();
    auto grad_tensor = grad_ctx->grad;
    auto op = grad_ctx->op;
    if (!op)
      continue;

    auto parents = op->getParents();
    auto gradients = op->backward(grad_tensor);

    // Accumulate gradients
    int i = 0;
    for (auto grad : gradients) {
      if (!grad) {
        i++;
        continue;
      }
      auto p = parents[i];
      auto p_storage = p->getStorage();
      p->accumulateGradient(grad, p_storage->getCudaContext());
      i++;
    }
    auto t_cuda_context = t->getStorage()->getCudaContext();
    if (grad_ctx->delGrad)
      t->freeGradient(t_cuda_context);
  }
}
