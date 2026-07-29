#pragma once
#include "tensor.cuh"
bool checkBroadcastable(Tensor A, Tensor target);
size_t calculateLogicalElementNo(std::array<size_t, MAX_DIMS> &shape,
                                 size_t rank);
std::vector<size_t> arrToVec(std::array<size_t, MAX_DIMS> arr, size_t rank);
bool checkContiguous(Tensor A);
void print(Tensor A);
