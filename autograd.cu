#include "tensor.cu"
#include <type_traits>

class Node;
using NodePtr = std::shared_ptr<Node>;

#define MAX_NODE_INPUTS 2

// Nodes

typedef enum {
  NODE_FLAG_NONE = 0,

  NODE_FLAG_REQUIRES_GRAD = (1 << 0),
  NODE_FLAG_PARAM = (1 << 1),
  NODE_FLAG_INPUT = (1 << 2),
  NODE_FLAG_OUTPUT = (1 << 3),
  NODE_FLAG_TARGET = (1 << 4),
  NODE_FLAG_COST = (1 << 5),
} NodeFlags;

typedef enum {
  NODE_OP_NULL = 0,
  NODE_OP_CREATE,
  __NODE_UNARY_OPS,
  NODE_OP_SCALE,
  NODE_OP_RELU,
  NODE_OP_SOFTMAX,
  NODE_OP_TANH,
  __NODE_BINARY_OPS,
  NODE_OP_ADD,
  NODE_OP_SUB,
  NODE_OP_MATMUL
} NodeOp;

#define NODE_NUM_INPUTS(op)                                                    \
  ((op) < __NODE_UNARY_OPS ? 0 : ((op) < __NODE_BINARY_OPS ? 1 : 2))

class Node : public std::enable_shared_from_this<Node> {
public:
  int index;
  int flags;

  Tensor value = nullptr;
  Tensor grad = nullptr;

  size_t numInputs = 0;
  std::shared_ptr<Node> inputs[MAX_NODE_INPUTS];
  NodeOp op;
};

// Creation API

NodePtr NodeCreate(Tensor input, int flags);
// NodePtr __NodeUnary(NodePtr A, int flags, NodeOp op);
// NodePtr __NodeBinary(NodePtr A, NodeOp B, int flags, NodeOp op);

NodePtr NodeScale(NodePtr A, float scalar, int flags);
NodePtr NodeRelu(NodePtr A, int flags);
NodePtr NodeTanh(NodePtr A, int flags);
NodePtr NodeSoftmax(NodePtr A, int flags);

NodePtr NodeAdd(NodePtr A, NodePtr B, int flags);
NodePtr NodeSub(NodePtr A, NodePtr B, int flags);
NodePtr NodeMatmul(NodePtr A, NodePtr B, int flags);

// Impl

NodePtr NodeCreate(Tensor input, int flags) {

  auto result = std::make_shared<Node>();
  result->value = input;
  result->flags = flags;

  return result;
}

NodePtr NodeScale(NodePtr A, float scalar, int flags) {
  auto value = A->value;

  // If parent requires gradient, this too requires gradient
  if (A->flags & NODE_FLAG_REQUIRES_GRAD) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }
  auto result = NodeCreate(TensorScale(value, scalar), flags);
  result->inputs[0] = A;
  result->op = NodeOp::NODE_OP_SCALE;
  result->numInputs = NODE_NUM_INPUTS(result->op);
  return result;
}

NodePtr NodeRelu(NodePtr A, int flags) {
  auto value = A->value;

  // If parent requires gradient, this too requires gradient
  if (A->flags & NODE_FLAG_REQUIRES_GRAD) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }
  auto result = NodeCreate(TensorRelu(value), flags);
  result->inputs[0] = A;
  result->op = NodeOp::NODE_OP_RELU;
  result->numInputs = NODE_NUM_INPUTS(result->op);
  return result;
}

NodePtr NodeTanh(NodePtr A, int flags) {
  auto value = A->value;

  // If parent requires gradient, this too requires gradient
  if (A->flags & NODE_FLAG_REQUIRES_GRAD) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }
  auto result = NodeCreate(TensorTanh(value), flags);
  result->inputs[0] = A;
  result->op = NodeOp::NODE_OP_TANH;
  result->numInputs = NODE_NUM_INPUTS(result->op);
  return result;
}

NodePtr NodeSoftmax(NodePtr A, int flags) {

  auto value = A->value;

  // If parent requires gradient, this too requires gradient
  if (A->flags & NODE_FLAG_REQUIRES_GRAD) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }
  auto result = NodeCreate(TensorSoftmax(value), flags);
  result->inputs[0] = A;
  result->op = NodeOp::NODE_OP_SOFTMAX;
  result->numInputs = NODE_NUM_INPUTS(result->op);
  return result;
}

NodePtr NodeAdd(NodePtr A, NodePtr B, int flags) {
  auto valueA = A->value;
  auto valueB = B->value;

  // If parent requires gradient, this too requires gradient
  if ((A->flags & NODE_FLAG_REQUIRES_GRAD) ||
      (B->flags & NODE_FLAG_REQUIRES_GRAD)) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }
  auto result = NodeCreate(TensorAdd(valueA, valueB), flags);
  result->inputs[0] = A;
  result->inputs[1] = B;
  result->op = NODE_OP_ADD;
  result->numInputs = NODE_NUM_INPUTS(result->op);
  return result;
}

NodePtr NodeSub(NodePtr A, NodePtr B, int flags) {
  auto valueA = A->value;
  auto valueB = B->value;

  // If parent requires gradient, this too requires gradient
  if ((A->flags & NODE_FLAG_REQUIRES_GRAD) ||
      (B->flags & NODE_FLAG_REQUIRES_GRAD)) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }
  auto result = NodeCreate(TensorSub(valueA, valueB), flags);
  result->inputs[0] = A;
  result->inputs[1] = B;
  result->op = NODE_OP_SUB;
  result->numInputs = NODE_NUM_INPUTS(result->op);
  return result;
}

NodePtr NodeMatmul(NodePtr A, NodePtr B, int flags) {
  auto valueA = A->value;
  auto valueB = B->value;

  // If parent requires gradient, this too requires gradient
  if ((A->flags & NODE_FLAG_REQUIRES_GRAD) ||
      (B->flags & NODE_FLAG_REQUIRES_GRAD)) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }
  auto result = NodeCreate(TensorMatmul(valueA, valueB), flags);
  result->inputs[0] = A;
  result->inputs[1] = B;
  result->op = NodeOp::NODE_OP_MATMUL;
  result->numInputs = NODE_NUM_INPUTS(result->op);
  return result;
}
