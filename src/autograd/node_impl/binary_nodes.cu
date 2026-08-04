#include <autograd.cuh>
#include <tensor.cuh>
#include <utils.cuh>

// NOTE: Forward Nodes

Node AddNode(Node A, Node B) {
  Tensor inA = A->data;
  Tensor inB = B->data;
  Tensor result = TensorAdd(inA, inB);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_ADD,
                                          A->requiresGrad || B->requiresGrad,
                                          A->retainsGrad || B->retainsGrad);
  r->inputs[0] = A;
  r->inputs[1] = B;
  return r;
}

Node SubNode(Node A, Node B) {
  Tensor inA = A->data;
  Tensor inB = B->data;
  Tensor result = TensorSub(inA, inB);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_SUB,
                                          A->requiresGrad || B->requiresGrad,
                                          A->retainsGrad || B->retainsGrad);
  r->inputs[0] = A;
  r->inputs[1] = B;
  return r;
}

Node MulNode(Node A, Node B) {
  Tensor inA = A->data;
  Tensor inB = B->data;
  Tensor result = TensorMul(inA, inB);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_SUB,
                                          A->requiresGrad || B->requiresGrad,
                                          A->retainsGrad || B->retainsGrad);
  r->inputs[0] = A;
  r->inputs[1] = B;
  return r;
}
Node MatmulNode(Node A, Node B);

// NOTE: Backward Nodes

bool __BinaryBackward__(Node node, Operation targetOp, Tensor &gA, Tensor &gB) {
  if (node->noInputs != 2 || node->op != targetOp)
    return false;

  gA = node->inputs[0]->grad;
  gB = node->inputs[1]->grad;
  if (!gA) {
    node->inputs[0]->allocGrad();
    gA = node->inputs[0]->grad;
  }
  if (!gB) {
    node->inputs[1]->allocGrad();
    gB = node->inputs[1]->grad;
  }
  return true;
}

void AddBackward(Node node) {
  Tensor gradA;
  Tensor gradB;
  Tensor topGrad = node->grad;

  if (!__BinaryBackward__(node, Operation::OP_ADD, gradA, gradB))
    return;
  node->inputs[0]->grad = TensorAdd(gradA, topGrad);
  node->inputs[1]->grad = TensorAdd(gradB, topGrad);
}

void SubBackward(Node node) {
  Tensor gradA;
  Tensor gradB;
  Tensor topGrad = node->grad;

  if (!__BinaryBackward__(node, Operation::OP_SUB, gradA, gradB))
    return;
  node->inputs[0]->grad = TensorAdd(gradA, topGrad);
  node->inputs[1]->grad = TensorAdd(gradB, TensorNeg(topGrad));
}

void MulBackward(Node node) {
  Tensor gradA;
  Tensor gradB;
  Tensor topGrad = node->grad;

  Tensor dataA = node->inputs[0]->data;
  Tensor dataB = node->inputs[1]->data;

  if (!__BinaryBackward__(node, Operation::OP_SUB, gradA, gradB))
    return;

  node->inputs[0]->grad = TensorAdd(gradA, TensorMul(topGrad, dataB));
  node->inputs[1]->grad = TensorAdd(gradB, TensorMul(topGrad, dataA));
}

void MatmulBackward(Node node) {}
