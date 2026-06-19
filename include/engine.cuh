#pragma once
#include "tensor.cuh"
#include <map>
#include <unordered_set>

class Engine {
public:
  void backward(const Tensor &root);

private:
  std::map<TensorObject *, uint> indeg_count;
  std::unordered_set<TensorObject *> visited;
  void count(TensorObject *root);
};
