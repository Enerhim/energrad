#pragma once
#include <tensor.cuh>

#define MAX_INPUTS 2

enum class Operation {
  NONE,
  _OP_UNARY_,
  OP_NEG,
  OP_ABS,
  OP_SQRT,
  OP_RECIPROCAL,
  OP_EXP,
  OP_NLOG,
  OP_LOG2,
  OP_LOG10,
  OP_SIN,
  OP_SINH,
  OP_COS,
  OP_COSH,
  OP_TANH,
  OP_FLOOR,
  OP_CEIL,
  OP_SCALE,
  OP_RELU,
  OP_SIGMOID,
  OP_SOFTMAX,
  _OP_BINARY_,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_MATMUL,

};

class AutogradNode : std::enable_shared_from_this<AutogradNode> {
public:
  AutogradNode(Tensor input, Operation op, bool requiresGrad, bool retainsGrad);

  std::shared_ptr<AutogradNode> inputs[MAX_INPUTS];
  size_t noInputs = 0;
  Operation op = Operation::NONE;

  bool requiresGrad = false;
  bool retainsGrad = false;

  Tensor data = nullptr;
  Tensor grad = nullptr;

  void allocGrad();
};

using Node = std::shared_ptr<AutogradNode>;

// NOTE: Forward API

Node CreateNode(Tensor input, bool requiresGrad, bool retainsGrad);

Node NegNode(Node A);
Node AbsNode(Node A);
Node SqrtNode(Node A);
Node ReciprocalNode(Node A);
Node ExpNode(Node A);
Node NLogNode(Node A);
Node Log2Node(Node A);
Node Log10Node(Node A);
Node SinNode(Node A);
Node SinhNode(Node A);
Node CosNode(Node A);
Node CoshNode(Node A);
Node TanhNode(Node A);
Node FloorNode(Node A);
Node CeilNode(Node A);
Node ScaleNode(Node A);
Node ReluNode(Node A);
Node SigmoidNode(Node A);
Node SoftmaxNode(Node A);

// Binary
Node AddNode(Node A, Node B);
Node SubNode(Node A, Node B);
Node MulNode(Node A, Node B);
Node MatmulNode(Node A, Node B);

// NOTE: Backward API
void NegBackward(Node node);
void AbsBackward(Node node);
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

// Binary
void AddBackward(Node node);
void SubBackward(Node node);
void MulBackward(Node node);
void MatmulBackward(Node node);
