#pragma once

#include <cstdint>

__global__ void fnv1a64(bool* match, uint64_t target, unsigned char** inp_strs, size_t inp_len);

__host__ uint64_t fnv1a64_cpu(const unsigned char* inp_str, const size_t inp_len);

__global__ void increment(unsigned char** inp_strs, size_t inp_len);