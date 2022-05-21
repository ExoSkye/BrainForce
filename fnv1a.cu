#include "fnv1a.cuh"

__constant__ const uint64_t FNV_PRIME = 1099511628211ull;
__constant__ const uint64_t OFFSET_BASIS = 14695981039346656037ull;

__global__ void increment(unsigned char** inp_strs, size_t inp_len) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 2; i < inp_len; i++) {
        inp_strs[id][i]++;
        if (inp_strs[id][i] != 0)
            break;
    }
}

__global__ void fnv1a64(
    bool* match, const uint64_t target, unsigned char** inp_strs, const size_t inp_len
) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned char* inp_str = inp_strs[id];

    inp_str[0] = blockIdx.x;
    inp_str[1] = threadIdx.x;

    uint64_t hash = OFFSET_BASIS;

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
}

__host__ uint64_t fnv1a64_cpu(const unsigned char* inp_str, const size_t inp_len) {
    uint64_t hash = OFFSET_BASIS;

    for (int i = 0; i < inp_len; i++) {
        unsigned char inp_char = inp_str[i];

        hash ^= inp_char;
        hash *= FNV_PRIME;
    }

    return hash;
}