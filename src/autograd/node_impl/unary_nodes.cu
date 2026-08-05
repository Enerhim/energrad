#include <autograd.cuh>
#include <cmath>
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
  r->ctx.params[0] = a;
  return r;
}

Node ReluNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorRelu(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_RELU,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node SigmoidNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorSigmoid(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_SIGMOID,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

Node SoftmaxNode(Node A) {
  Tensor inA = A->data;
  Tensor result = TensorSoftmax(inA);

  Node r = std::make_shared<AutogradNode>(result, Operation::OP_SOFTMAX,
                                          A->requiresGrad, A->retainsGrad);
  r->inputs[0] = A;
  return r;
}

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
  Tensor topGrad = node->grad;

  if (!__UnaryBackward__(node, Operation::OP_NEG, gradA))
    return;
  node->inputs[0]->grad =
      TensorAdd(gradA, TensorInit(gradA->shape, gradA->rank, -1.0f,
                                  gradA->storage->device, true));
}

void AbsBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataA = node->inputs[0]->data;

  if (!__UnaryBackward__(node, Operation::OP_ABS, gradA))
    return;
  node->inputs[0]->grad = TensorAdd(gradA, TensorAbsBackward(dataA));
}

void SqrtBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataA = node->inputs[0]->data;

  if (!__UnaryBackward__(node, Operation::OP_SQRT, gradA))
    return;

  auto halfTensor =
      TensorInit(gradA->shape, gradA->rank, 0.5f, gradA->storage->device, true);

  node->inputs[0]->grad = TensorAdd(
      gradA,
      TensorMul(TensorMul(halfTensor, TensorReciprocal(TensorSqrt(dataA))),
                topGrad));
}

void ReciprocalBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataCurrent = node->data;

  if (!__UnaryBackward__(node, Operation::OP_RECIPROCAL, gradA))
    return;

  node->inputs[0]->grad =
      TensorAdd(gradA, TensorMul(TensorNeg(topGrad),
                                 TensorMul(dataCurrent, dataCurrent)));
}

void ExpBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataCurrent = node->data;

  if (!__UnaryBackward__(node, Operation::OP_EXP, gradA))
    return;

  node->inputs[0]->grad = TensorAdd(gradA, TensorMul(topGrad, dataCurrent));
}

void NLogBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataA = node->inputs[0]->data;

  if (!__UnaryBackward__(node, Operation::OP_NLOG, gradA))
    return;

  node->inputs[0]->grad =
      TensorAdd(gradA, TensorMul(topGrad, TensorReciprocal(dataA)));
}

void Log2Backward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataA = node->inputs[0]->data;

  if (!__UnaryBackward__(node, Operation::OP_LOG2, gradA))
    return;

  auto ln2 = TensorInit(gradA->shape, gradA->rank, logf(2.0f),
                        gradA->storage->device, true);

  node->inputs[0]->grad = TensorAdd(
      gradA, TensorMul(topGrad, TensorReciprocal(TensorMul(ln2, dataA))));
}

void Log10Backward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataA = node->inputs[0]->data;

  if (!__UnaryBackward__(node, Operation::OP_LOG10, gradA))
    return;

  auto ln10 = TensorInit(gradA->shape, gradA->rank, logf(10.0f),
                         gradA->storage->device, true);

  node->inputs[0]->grad = TensorAdd(
      gradA, TensorMul(topGrad, TensorReciprocal(TensorMul(ln10, dataA))));
}

void SinBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataA = node->inputs[0]->data;

  if (!__UnaryBackward__(node, Operation::OP_SIN, gradA))
    return;

  node->inputs[0]->grad =
      TensorAdd(gradA, TensorMul(topGrad, TensorCos(dataA)));
}

void SinhBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataA = node->inputs[0]->data;

  if (!__UnaryBackward__(node, Operation::OP_SINH, gradA))
    return;

  node->inputs[0]->grad =
      TensorAdd(gradA, TensorMul(topGrad, TensorCosh(dataA)));
}
void CosBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataA = node->inputs[0]->data;

  if (!__UnaryBackward__(node, Operation::OP_COS, gradA))
    return;

  node->inputs[0]->grad =
      TensorAdd(gradA, TensorMul(TensorNeg(topGrad), TensorSin(dataA)));
}
void CoshBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataA = node->inputs[0]->data;

  if (!__UnaryBackward__(node, Operation::OP_COSH, gradA))
    return;

  node->inputs[0]->grad =
      TensorAdd(gradA, TensorMul(topGrad, TensorSinh(dataA)));
}

void TanhBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;
  Tensor dataCurrent = node->data;

  if (!__UnaryBackward__(node, Operation::OP_TANH, gradA))
    return;

  auto ones =
      TensorInit(gradA->shape, gradA->rank, 1.0f, gradA->storage->device, true);

  node->inputs[0]->grad = TensorAdd(
      gradA,
      TensorMul(topGrad, TensorSub(ones, TensorMul(dataCurrent, dataCurrent))));
}

void FloorBackward(Node node) {
  // No op essentially
}

void CeilBackward(Node node) {
  // No op essentially
}
void ScaleBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;

  if (!__UnaryBackward__(node, Operation::OP_SCALE, gradA))
    return;

  auto param = TensorInit(gradA->shape, gradA->rank, node->ctx.params[0],
                          gradA->storage->device, true);

  node->inputs[0]->grad = TensorAdd(gradA, TensorMul(topGrad, param));
}

void ReluBackward(Node node) {
  Tensor gradA;
  Tensor topGrad = node->grad;

  if (!__UnaryBackward__(node, Operation::OP_RELU, gradA))
    return;

  node->inputs[0]->grad =
      TensorAdd(gradA, TensorMul(topGrad, TensorReluBackward(gradA)));
}
void SigmoidBackward(Node node);
void SoftmaxBackward(Node node);
