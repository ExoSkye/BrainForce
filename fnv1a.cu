#include "fnv1a.cuh"
#include "128-bit-literals/suffix128.hpp"

__constant__ const __uint128_t FNV_PRIME = 309485009821345068724781371_128;
__constant__ const __uint128_t OFFSET_BASIS = 144066263297769815596495629667062367629_128;

__global__ void fnv1a64(
    bool* match, const __uint128_t target, unsigned char** inp_strs, const size_t inp_len
) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned char* inp_str = inp_strs[id];

    inp_str[0] = blockIdx.x;
    inp_str[1] = threadIdx.x;

    __uint128_t hash = OFFSET_BASIS;

    for (int i = 0; i < inp_len; i++) {
        unsigned char inp_char = inp_str[i];

        hash ^= inp_char;
        hash *= FNV_PRIME;
    }

    if (hash == target) {
        match[id] |= true;
    }
    else {
        match[id] |= false;
    }

    for (int i = 2; i < inp_len; i++) {
        inp_strs[id][i]++;
        if (inp_strs[id][i] != 0)
            break;
    }
}

__host__ __uint128_t fnv1a64_cpu(const unsigned char* inp_str, const size_t inp_len) {
    __uint128_t hash = OFFSET_BASIS;

    for (int i = 0; i < inp_len; i++) {
        unsigned char inp_char = inp_str[i];

        hash ^= inp_char;
        hash *= FNV_PRIME;
    }

    return hash;
}
