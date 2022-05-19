cmake_minimum_required(VERSION 3.22)
project(BrainForce CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

add_executable(BrainForce main.cu adler.cu adler.cuh)

set_target_properties(BrainForce PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
