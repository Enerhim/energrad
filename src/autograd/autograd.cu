#include <autograd.cuh>
#include <tensor.cuh>

AutogradNode::AutogradNode(Tensor input, Operation op, bool requiresGrad,
                           bool retainsGrad)
    : data(input), op(op), requiresGrad(requiresGrad),
      retainsGrad(retainsGrad) {

  noInputs =
      op > Operation::_OP_UNARY_ ? (op > Operation::_OP_BINARY_ ? 2 : 1) : 0;
}

void AutogradNode::allocGrad() {
  grad = TensorInit(data->shape, data->rank, 0.0f, data->storage->device, true);
}

Node CreateNode(Tensor input, bool requiresGrad, bool retainsGrad) {
  Node n = std::make_shared<AutogradNode>(input, Operation::NONE, requiresGrad,
                                          retainsGrad);
  return n;
}
