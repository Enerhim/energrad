#include "tensor.hpp"
#include <iostream>

int main() {
  std::vector<TensorPtr> X(100000);

  TensorPtr sum = std::make_shared<Tensor>("Sum", 100, 100, false, 0.0f);
  for (int i = 0; i < 100000; i++) {
    X[i] = std::make_shared<Tensor>("X" + std::to_string(i), 100, 100, false,
                                    0.0f, 1.0f);

    sum = sum + X[i];
    if (i % 1000 == 0)
      std::cout << "Sum It: " << i << std::endl;
  }

  sum = sum / std::make_shared<Tensor>("Div", 28, 28, false, 10000.0);

  sum->pData();

  return 0;
}
