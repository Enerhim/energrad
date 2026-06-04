#include "tensor.hpp"

int main() {

  TensorPtr a = std::make_shared<Tensor>("a", 5, 5, true, -1.0f, 1.0f);
  TensorPtr b = std::make_shared<Tensor>("b", 5, 5, true, -1.0f, 1.0f);

  TensorPtr d = std::make_shared<Tensor>("d", 5, 5, true, -1.0f, 1.0f);
  TensorPtr e = std::make_shared<Tensor>("e", 5, 5, true, -1.0f, 1.0f);

  auto c = a * b;
  c->setLabel("c");
  auto f = d * e;
  f->setLabel("f");

  auto g = c + f;
  g->setLabel("g");

  // Seeding
  g->cudaGradFill(1.0f);
  g->backward();
  c->backward();
  f->backward();
  g->pGrads();
  c->pGrads();
  f->pGrads();

  a->pGrads();
  b->pGrads();
  d->pGrads();
  e->pGrads();
  // .backward() takes the top gradient, adds its local part and accumulates it
  // to the parents' gradients. Accumulate because parents may recieve dicks
  // from every child
  //
  // g->backward();
  // Now g should yeet dg to c and f directly
  // c->backward();
  // Now c will yeet a * dc to b and b * dc to a
  // f->backward();
  // Now f will yeet d * df to e and e * df to d

  return 0;
}
