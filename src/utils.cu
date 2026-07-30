#include <tensor.cuh>
#include <utils.cuh>

// NOTE:  Utils

std::vector<size_t> arrToVec(std::array<size_t, MAX_DIMS> arr, size_t rank) {
  std::vector<size_t> vec(rank);
  std::copy(arr.begin(), arr.begin() + rank, vec.begin());
  return vec;
}

bool checkBroadcastable(Tensor A, Tensor target) {
  auto shapeA = A->shape, shapeTarget = target->shape;
  auto rankA = A->rank, rankTarget = target->rank;

  if (rankA > rankTarget)
    return false;

  int i = static_cast<int>(rankTarget) - 1, j = static_cast<int>(rankA) - 1;
  while (j >= 0) {
    if (shapeA[j] != shapeTarget[i] && shapeA[j] != 1)
      return false;

    --i, --j;
  }

  return true;
}

size_t calculateLogicalElementNo(std::array<size_t, MAX_DIMS> &shape,
                                 size_t rank) {
  size_t elements = 1;
  for (int i = 0; i < rank; i++) {
    elements *= shape[i];
  }
  return elements;
}

bool checkContiguous(Tensor A) {
  size_t rank = A->rank;
  if (rank == 0)
    return true;

  size_t currentStride = 1;
  for (int dim = static_cast<int>(rank) - 1; dim >= 0; dim--) {
    if (A->strides[dim] != currentStride)
      return false;
    currentStride *= A->shape[dim];
  }

  return true;
}

void print(Tensor A) {
  auto shape = A->shape;
  auto strides = A->strides;
  auto rank = A->rank;
  auto data = A->storage->data;
  auto elements = calculateLogicalElementNo(shape, rank);

  std::cout << "[ ";
  for (int idx = 0; idx < elements; idx++) {
    size_t offset = 0;
    size_t remaining = idx;

    for (int dim = static_cast<int>(rank) - 1; dim >= 0; dim--) {
      size_t indice = remaining % shape[dim];
      remaining /= shape[dim];

      offset += indice * strides[dim];
    }
    std::cout << data[offset] << " ";
  }
  std::cout << " ]" << std::endl;
}
