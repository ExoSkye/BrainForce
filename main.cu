#include <cstdio>
#include <cstdint>
#include <ctime>

#include "adler.cuh"

const uint64_t LEN = 4;

uint32_t adler_cpu(const unsigned char* inp_str, const size_t inp_len) {
    uint16_t a = 1;
    uint16_t b = 0;

    for (int i = 0; i < inp_len; i++) {
        const unsigned char inp_char = inp_str[i];
        a += (uint8_t) inp_char;
        a %= 65521;

        b += a;
        b %= 65521;
    }

    uint32_t out = b << 16 | a;

    return out;
}

int main() {
    auto** bruteforce_bytes = (unsigned char**)malloc(sizeof(unsigned char*) * 65536);
    auto* zero_bytes = (unsigned char*)calloc(sizeof(unsigned char), LEN);

    unsigned char** d_bruteforce_bytes = nullptr;

    cudaError_t err;

    err = cudaMalloc((void**)&d_bruteforce_bytes, sizeof(unsigned char*) * 65536);
    if (err != cudaSuccess) {
        printf("Failed to initialize, exiting...\n");
        exit(1);
    }

    for (int i = 0; i < 65536; i++) {
        err = cudaMalloc((void**)&bruteforce_bytes[i], sizeof(unsigned char) * LEN);
        if (err != cudaSuccess) {
            printf("Failed to initialize, exiting...\n");
            exit(1);
        }
        err = cudaMemcpy(bruteforce_bytes[i], zero_bytes, sizeof(unsigned char) * LEN, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Failed to initialize, exiting...\n");
            exit(1);
        }
    }

    err = cudaMemcpy(d_bruteforce_bytes, bruteforce_bytes, sizeof(unsigned char*) * 65536,
                           cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        printf("Failed to initialize, exiting...\n");
        exit(1);
    }

    auto* matches = (bool*)calloc(sizeof(bool), 65536);

    bool* d_matches = NULL;

    err = cudaMalloc((void**)&d_matches, sizeof(bool) * 65536);

    if (err != cudaSuccess) {
        printf("Failed to initialize, exiting...\n");
        exit(0);
    }

    err = cudaMemcpy(d_matches, matches, sizeof(bool) * 65536, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        printf("Failed to initialize, exiting...\n");
        exit(0);
    }

    const uint32_t target = adler_cpu((const unsigned char*)("a"), 1);

    bool found = false;

    for (int len = 1; len < LEN + 1 && !found; len++) {
        printf("Trying length: %i\n", len);

        int iter = 0;

        double possible_combos = pow(2, 8 * len);

        printf("%f possible combinations\n", possible_combos);

        uint64_t combos = 0;

        time_t start_time = time(NULL);

        while (!found) {
            adler<<<256, 256>>>(d_matches, target, d_bruteforce_bytes, len);
            increment<<<256, 256>>>(d_bruteforce_bytes, len);

            if (iter % 10000 == 5000 || combos == possible_combos) {
                time_t cur_time = time(NULL) - start_time;
                time_t time_left = ((double) cur_time / (double) combos) * (possible_combos - (double) combos);
                char* time_str = (char*) calloc(sizeof(char), 10);
                struct tm* timeinfo = localtime(&time_left);

                strftime(time_str, 10, "%X", timeinfo);
                printf("Checking results (%f%% done - estimated time: %s)\n", (double) combos / possible_combos * 100,
                       time_str);
                cudaMemcpy(matches, d_matches, sizeof(bool) * 65536, cudaMemcpyDeviceToHost);

                for (int i = 0; i < 65536; i++) {
                    if (matches[i]) {
                        printf("FOUND MATCH ON THREAD %d\n", i);

                        cudaMemcpy(zero_bytes,
                                   bruteforce_bytes[i],
                                   sizeof(unsigned char) * len,
                                   cudaMemcpyDeviceToHost);

                        unsigned char* cpy_bytes = zero_bytes;

                        while(*cpy_bytes)
                            printf("%02x", (unsigned int) *cpy_bytes++);
                        printf("\n");

                        found = true;
                        break;
                    }
                }
            }

            iter++;


            if ((double) combos > possible_combos) {
                printf("Couldn't find that value in given length space, maybe try a different length?\n");
                break;
            }

            combos += std::min(pow(2, 8 * len), 65536.0);
        }
    }

    for (int i = 0; i < 65536; i++) {
        cudaFree(bruteforce_bytes[i]);
    }

    cudaFree(d_bruteforce_bytes);
    cudaFree(d_matches);

    free(bruteforce_bytes);
    free(matches);
    free(zero_bytes);

    return 0;
}
