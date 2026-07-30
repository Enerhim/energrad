cmake_minimum_required(VERSION 3.20)

project(energrad LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENERGRAD_BUILD_TESTS "Build energrad tests" ON)

if (ENERGRAD_BUILD_TESTS)
  enable_testing()

  add_executable(ops_tests
      tests/ops.cu
  )

  target_link_libraries(ops_tests
      PRIVATE
          energrad
  )

  set_target_properties(ops_tests PROPERTIES
      CUDA_ARCHITECTURES native
  )

  add_test(NAME ops_tests COMMAND ops_tests)
endif()

file(GLOB_RECURSE ENERGRAD_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(energrad ${ENERGRAD_SOURCES})

target_include_directories(energrad
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(build main.cu)

target_link_libraries(build
    PRIVATE
        energrad
)

set_target_properties(energrad build PROPERTIES
    CUDA_ARCHITECTURES native
)
