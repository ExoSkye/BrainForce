cmake_minimum_required(VERSION 3.22)
project(BrainForce C CXX CUDA)

find_package(Boost REQUIRED COMPONENTS ALL)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(BrainForce main.cu fnv1a.cu fnv1a.cuh)
target_link_libraries(BrainForce PRIVATE Boost::boost Boost::program_options)

# Weird hack because my system does weird shit
# target_include_directories(BrainForce PRIVATE BEFORE SYSTEM /usr/lib/gcc/x86_64-pc-linux-gnu/11.3.0/include/ /usr/lib/gcc/x86_64-pc-linux-gnu/11.3.0/include/c++/)


set_target_properties(BrainForce PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
