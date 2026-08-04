#include <autograd.cuh>

void NegBackward(Node node) {}
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
