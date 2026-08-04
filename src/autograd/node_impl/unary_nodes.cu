#include <autograd.cuh>
#include <tensor.cuh>
#include <utils.cuh>

// NOTE: Forward Nodes

Node NegNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorNeg(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_NEG,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node AbsNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorAbs(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_ABS,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node SqrtNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorSqrt(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_SQRT,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node ReciprocalNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorReciprocal(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_RECIPROCAL,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node ExpNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorExp(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_EXP,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node NLogNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorNLog(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_NLOG,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node Log2Node(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorLog2(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_LOG2,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node Log10Node(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorLog10(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_LOG10,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node SinNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorSin(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_SIN,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node SinhNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorSinh(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_SINH,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}
Node CosNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorCos(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_COS,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}
Node CoshNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorCosh(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_COSH,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node TanhNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorTanh(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_TANH,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}
Node FloorNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorFloor(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_FLOOR,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}
Node CeilNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorCeil(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_CEIL,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}
Node ScaleNode(Node A, float a) {
  Tensor inA = A->data;
  Tensor result = TensorScale(inA, a);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_SCALE,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

// All special ones
Node ReluNode(Node A);
Node SigmoidNode(Node A);
Node SoftmaxNode(Node A);

// NOTE: Backward Nodes

bool __UnaryBackward__(Node node, Operation op, Tensor &gA) {
  if (node->noInputs != 1 || node->op != op) {
    return false;
  }
  gA = node->inputs[0]->grad;
  if (!gA) {
    node->inputs[0]->allocGrad();
    gA = node->inputs[0]->grad;
  }
  return true;
}

void NegBackward(Node node) {
  Tensor gradA;
  Tensor gradB;
  Tensor topGrad = node->grad;

  if (!__UnaryBackward__(node, Operation::OP_NEG, gradA))
    return;
  node->inputs[0]->grad =
      TensorAdd(gradA, TensorInit(gradA->shape, gradA->rank, -1.0f,
                                  gradA->storage->device, true));
}

// ABS Requires custom kernel;
void AbsBackward(Node node) {
  Tensor gradA;
  Tensor gradB;
  Tensor topGrad = node->grad;

  if (!__UnaryBackward__(node, Operation::OP_NEG, gradA))
    return;
  node->inputs[0]->grad =
      TensorAdd(gradA, TensorInit(gradA->shape, gradA->rank, -1.0f,
                                  gradA->storage->device, true));
}

void SqrtBackward(Node node);
void ReciprocalBackward(Node node);
void ExpBackward(Node node);
void NLogBackward(Node node);
void Log2Backward(Node node);
void Log10Backward(Node node);
void SinBackward(Node node);
void SinhBackward(Node node);
void CosBackward(Node node);
void CoshBackward(Node node);
void TanhBackward(Node node);
void FloorBackward(Node node);
void CeilBackward(Node node);
void ScaleBackward(Node node);
void ReluBackward(Node node);
void SigmoidBackward(Node node);
void SoftmaxBackward(Node node);
