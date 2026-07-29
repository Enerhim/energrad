# Energrad 

An autograd engine in CUDA

### Project Structure

```
.
├── CMakeLists.txt
├── include
│   ├── tensor.cuh
│   └── utils.cuh
├── main.cu
├── README.md
└── src
    ├── tensor
    │   └── ops
    │       ├── binary_ops.cu
    │       ├── io_init_ops.cu
    │       ├── shape_ops.cu
    │       └── unary_ops.cu
    ├── tensor.cu
    └── utils.cu


```
### Build

`git clone https://github.com/Enerhim/energrad.gir .`

`cmake -S . -B build -G Ninja`

`cmake --build build -j`
