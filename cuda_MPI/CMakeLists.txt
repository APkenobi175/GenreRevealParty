cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Project Name
project(GenereRevealMPICuda CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)


# Find MPI
find_package(MPI REQUIRED)



add_library(kernels
  kernels.cu
)

target_compile_features(kernels PUBLIC cxx_std_11)
set_target_properties(kernels
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(mpi-cuda mpi_main.cpp)

target_link_libraries(mpi-cuda PRIVATE
    kernels
    MPI::MPI_CXX
)

target_include_directories(mpi-cuda
    PUBLIC "../include"
    PRIVATE ${MPI_CXX_INCLUDE_DIRS}
)
