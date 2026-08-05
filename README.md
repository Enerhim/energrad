# Energrad 

An autograd engine in CUDA

Currently supports: 
- Math operations on n-dim tensors
- Separated tensor views and tensor storage
- Autograd graph building based on tensor operations
- Broadcasting, transpose and other shape ops
- PCG RNG for random number generation

### Project Structure

```
.
├── CMakeLists.txt
├── include
│   ├── autograd.cuh
│   ├── pcg
│   │   ├── pcg_extras.hpp
│   │   ├── pcg_random.hpp
│   │   └── pcg_uint128.hpp
│   ├── tensor.cuh
│   └── utils.cuh
├── main.cu
├── README.md
├── src
│   ├── autograd
│   │   ├── autograd.cu
│   │   └── node_impl
│   │       ├── binary_nodes.cu
│   │       ├── shape_nodes.cu
│   │       └── unary_nodes.cu
│   ├── tensor
│   │   ├── ops
│   │   │   ├── binary_ops.cu
│   │   │   ├── io_init_ops.cu
│   │   │   ├── shape_ops.cu
│   │   │   └── unary_ops.cu
│   │   └── tensor.cu
│   └── utils.cu
└── tests
    └── op_tests.cu
```
### Build

```bash
git clone https://github.com/Enerhim/energrad.git 
cmake -S . -B build -G Ninja
cmake --build build -j
```

