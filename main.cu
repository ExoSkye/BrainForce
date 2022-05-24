#include <cstdio>
#include <cstdint>
#include <ctime>
#include <cinttypes>
#include <fstream>

#include "fnv1a.cuh"
#include <boost/program_options.hpp>
#include <iostream>

//const uint64_t LEN = 5;

namespace po = boost::program_options;

int main(int argc, char** argv) {

    po::options_description desc("Parameters");
    desc.add_options()
        ("help", "Print a help message")
        ("target", po::value<std::string>(), "The hash to find (required)")
        ("length", po::value<uint64_t>(), "The max length of the string the hash can represent (required)")
        ("precompute", po::value<float>(), "Number of gigabytes of bruteforce inputs to precompute (required)")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
    }

    if (!vm.count("target") || !vm.count("length") || !vm.count("precompute")) {
        std::cout << "Required parameters are not set, exiting..." << std::endl;
        exit(1);
    }

    uint64_t LEN = vm["length"].as<uint64_t>();
    std::string target_std_string = vm["target"].as<std::string>();
    const char* target_str = target_std_string.c_str();
    uint64_t precompute = vm["precompute"].as<float>() * 1073741824;

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

    const __uint128_t target = fnv1a64_cpu((const unsigned char*)target_str, LEN);

    bool found = false;

    char* time_str = (char*) calloc(sizeof(char), 10);

    for (int len = 1; len < LEN + 1 && !found; len++) {
        printf("Trying length: %i\n", len);

        int iter = 0;

        double possible_combos = pow(2, 8 * len);

        printf("%f possible combinations\n", possible_combos);

        uint64_t combos = 0;

        time_t start_time = time(NULL);

        while (!found) {
            fnv1a64<<<256, 256>>>(d_matches, target, d_bruteforce_bytes, len);
            
            if (iter % 10000 == 5000 || combos == possible_combos) {
                time_t cur_time = time(NULL) - start_time;
                time_t time_left = ((double) cur_time / (double) combos) * (possible_combos - (double) combos);
                struct tm* timeinfo = localtime(&time_left);

                strftime(time_str, 10, "%X", timeinfo);
                printf("Checking results (%f%% done - estimated time: %s - %f H/s)\n", (double) combos /
                    possible_combos * 100, time_str, (float) combos / (float) cur_time);
                cudaMemcpy(matches, d_matches, sizeof(bool) * 65536, cudaMemcpyDeviceToHost);

                for (int i = 0; i < 65536; i++) {
                    if (matches[i]) {
                        printf("FOUND MATCH ON THREAD %d\n", i);

                        cudaMemcpy(zero_bytes,
                                   bruteforce_bytes[i],
                                   sizeof(unsigned char) * len,
                                   cudaMemcpyDeviceToHost);

                        unsigned char* cpy_bytes = zero_bytes;

                        while(cpy_bytes < zero_bytes + LEN)
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
    free(time_str);

    return 0;
}
