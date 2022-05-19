#pragma once
#include <cstdint>

__global__ void increment(unsigned char** inp_strs, size_t inp_len);

__global__ void adler(bool* match, uint32_t target, unsigned char** inp_strs, size_t inp_len);