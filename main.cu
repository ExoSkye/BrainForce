#include <cstdio>
#include <cstdint>
#include <ctime>

#include "adler.cuh"

const uint64_t LEN = 4;

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

    int iter = 0;

    double possible_combos = pow(2, 8*LEN);

    printf("%f possible combinations\n", possible_combos);

    uint64_t combos = 0;

    time_t start_time = time(NULL);

    bool found = false;

    while (!found) {
        adler<<<256, 256>>>(d_matches, 0xDEADBEEF, d_bruteforce_bytes, LEN);
        increment<<<256, 256>>>(d_bruteforce_bytes, LEN);

        if (iter % 10000 == 5000) {
            time_t cur_time = time(NULL) - start_time;
            time_t time_left = ((double)cur_time / (double)combos) * (possible_combos - (double)combos);
            char* time_str = (char*)calloc(sizeof(char), 10);
            struct tm* timeinfo = localtime(&time_left);

            strftime(time_str, 10, "%X", timeinfo);
            printf("Checking results (%f%% done - estimated time: %s)\n", (double)combos / possible_combos * 100,
                   time_str);
            cudaMemcpy(matches, d_matches, sizeof(bool) * 65536, cudaMemcpyDeviceToHost);

            for (int i = 0; i < 65536; i++) {
                if (matches[i]) {
                    printf("FOUND MATCH ON THREAD %d\n", i);

                    cudaMemcpy(zero_bytes, bruteforce_bytes[i], sizeof(unsigned char) * LEN, cudaMemcpyDeviceToHost);
                    printf("%s", zero_bytes);

                    found = true;
                    break;
                }
            }
        }

        iter++;

        combos += 65536;

        if ((double)combos > possible_combos) {
            printf("Couldn't find that value in given length space, maybe try a different length?");
            break;
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
