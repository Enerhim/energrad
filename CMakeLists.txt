cmake_minimum_required(VERSION 3.20)

project(energrad LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(ENERGRAD_BUILD_TESTS "Build energrad tests" ON)
option(ENERGRAD_BUILD_EXAMPLES "Build the energrad example executable" ON)

# --- Library ---------------------------------------------------------------

file(GLOB_RECURSE ENERGRAD_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(energrad ${ENERGRAD_SOURCES})

target_include_directories(energrad
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(energrad PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)

# --- Example ---------------------------------------------------------------

if (ENERGRAD_BUILD_EXAMPLES)
  add_executable(example main.cu)

  target_link_libraries(example
      PRIVATE
          energrad
  )

  set_target_properties(example PROPERTIES
      CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  )
endif()

# --- Tests -----------------------------------------------------------------

if (ENERGRAD_BUILD_TESTS)
  enable_testing()

  add_executable(ops_tests
      tests/op_tests.cu
  )

  target_link_libraries(ops_tests
      PRIVATE
          energrad
  )

  set_target_properties(ops_tests PROPERTIES
      CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  )

  add_test(NAME ops_tests COMMAND ops_tests)

  set_tests_properties(ops_tests PROPERTIES
      TIMEOUT 120
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()
