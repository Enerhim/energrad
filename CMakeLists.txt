cmake_minimum_required(VERSION 4.0)
project(energrad LANGUAGES CXX CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp")

add_executable(main main.cu ${SOURCES})
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
