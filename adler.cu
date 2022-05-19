#include "adler.cuh"

#include <cstdint>
#include <cstdio>

__global__ void increment(unsigned char** inp_strs, size_t inp_len) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 2; i < inp_len; i++) {
        inp_strs[id][i]++;
        if (inp_strs[id][i] != 0)
            break;
    }
}

__global__ void adler(bool* match, const uint32_t target, unsigned char** inp_strs, const size_t inp_len) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned char* inp_str = inp_strs[id];

    inp_str[0] = blockIdx.x;
    inp_str[1] = threadIdx.x;

    uint16_t a = 1;
    uint16_t b = 0;

    for (int i = 0; i < inp_len; i++) {
        unsigned char inp_char = inp_str[i];
        a += (uint8_t)inp_char;
        a %= 65521;

        b += a;
        b %= 65521;
    }

    uint32_t out = b << 16 | a;

    if (out == target) {
        match[id] |= true;
    }
    else {
        match[id] |= false;
    }
}