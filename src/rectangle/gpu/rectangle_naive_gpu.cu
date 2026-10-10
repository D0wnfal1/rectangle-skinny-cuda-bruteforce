#include <iostream>
#include <cstdint>
#include <iomanip>
#include <cmath>

__device__ int d_found = 0;
__device__ uint32_t d_found_key = 0;

__constant__ uint8_t RECTANGLE_SBOX[16] = {
    0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9, 0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2
};

__constant__ uint8_t RECTANGLE_RC[25] = {
    0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16, 0x0C, 0x19, 0x13, 0x07, 0x0F,
    0x1F, 0x1E, 0x1D, 0x1A, 0x15, 0x0A, 0x14, 0x08, 0x11, 0x03, 0x06, 0x0D
};

__device__ void rectangle_key_schedule(uint16_t master_key[5], uint16_t round_keys[26][4]) {
    uint16_t v[5] = { master_key[0], master_key[1], master_key[2], master_key[3], master_key[4] };
    for (int i = 0; i < 26; i++) {
        round_keys[i][0] = v[0];
        round_keys[i][1] = v[1];
        round_keys[i][2] = v[2];
        round_keys[i][3] = v[3];
        if (i == 25) break;

        for (int col = 0; col < 4; col++) {
            uint8_t nibble = ((v[3] >> col) & 1) << 3 | ((v[2] >> col) & 1) << 2 |
                ((v[1] >> col) & 1) << 1 | ((v[0] >> col) & 1);
            uint8_t s = RECTANGLE_SBOX[nibble];
            v[0] = (v[0] & ~(1 << col)) | ((s & 1) << col);
            v[1] = (v[1] & ~(1 << col)) | (((s >> 1) & 1) << col);
            v[2] = (v[2] & ~(1 << col)) | (((s >> 2) & 1) << col);
            v[3] = (v[3] & ~(1 << col)) | (((s >> 3) & 1) << col);
        }

        uint16_t r0_old = v[0];
        v[0] = ((v[0] << 8) | (v[0] >> 8)) ^ v[1];
        v[1] = v[2];
        v[2] = v[3];
        v[3] = ((v[3] << 12) | (v[3] >> 4)) ^ v[4];
        v[4] = r0_old;

        v[0] ^= RECTANGLE_RC[i];
    }
}

__device__ void rectangle_encrypt_naive(uint16_t state[4], const uint16_t round_keys[26][4]) {
    for (int rnd = 0; rnd < 25; ++rnd) {
        state[0] ^= round_keys[rnd][0];
        state[1] ^= round_keys[rnd][1];
        state[2] ^= round_keys[rnd][2];
        state[3] ^= round_keys[rnd][3];

        uint16_t next_state[4] = { 0, 0, 0, 0 };
        for (int col = 0; col < 16; ++col) {
            uint8_t n = ((state[3] >> col) & 1) << 3 | ((state[2] >> col) & 1) << 2 |
                ((state[1] >> col) & 1) << 1 | ((state[0] >> col) & 1);
            uint8_t s = RECTANGLE_SBOX[n];
            next_state[0] |= ((s & 1) << col);
            next_state[1] |= (((s >> 1) & 1) << col);
            next_state[2] |= (((s >> 2) & 1) << col);
            next_state[3] |= (((s >> 3) & 1) << col);
        }
        state[0] = next_state[0]; state[1] = next_state[1];
        state[2] = next_state[2]; state[3] = next_state[3];
        state[1] = (state[1] << 1) | (state[1] >> 15);
        state[2] = (state[2] << 12) | (state[2] >> 4);
        state[3] = (state[3] << 13) | (state[3] >> 3);
    }
    state[0] ^= round_keys[25][0];
    state[1] ^= round_keys[25][1];
    state[2] ^= round_keys[25][2];
    state[3] ^= round_keys[25][3];
}

__global__ void rectangle_bruteforce_naive_kernel(uint16_t pt0, uint16_t pt1, uint16_t pt2, uint16_t pt3,
    uint16_t ct0, uint16_t ct1, uint16_t ct2, uint16_t ct3) {
    if (d_found) return;
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    uint16_t master_key[5] = { (uint16_t)(thread_id & 0xFFFF), (uint16_t)((thread_id >> 16) & 0xFFFF), 0, 0, 0 };
    uint16_t round_keys[26][4];
    rectangle_key_schedule(master_key, round_keys);

    uint16_t state[4] = { pt0, pt1, pt2, pt3 };
    rectangle_encrypt_naive(state, round_keys);

    if (state[0] == ct0 && state[1] == ct1 && state[2] == ct2 && state[3] == ct3) {
        if (atomicExch(&d_found, 1) == 0) d_found_key = thread_id;
    }
}

int main() {
    uint16_t pt[4] = { 0x1122, 0x3344, 0x5566, 0x7788 };
    uint16_t ct[4] = { 0xABCD, 0xEF01, 0x2345, 0x6789 };

    int power = 16;

    uint64_t total_keys = 1ULL << power;
    int threads_per_block = 256;
    unsigned int blocks_per_grid = (unsigned int)(total_keys / threads_per_block);
    if (blocks_per_grid == 0) blocks_per_grid = 1;

    std::cout << " RECTANGLE-80 GPU CUDA Naive\n";
    std::cout << " Key Space     : 2^" << power << "\n";
    std::cout << " Configuration : Grid " << blocks_per_grid << " blocks, " << threads_per_block << " threads\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    rectangle_bruteforce_naive_kernel << <blocks_per_grid, threads_per_block >> > (
        pt[0], pt[1], pt[2], pt[3], ct[0], ct[1], ct[2], ct[3]);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double elapsed_sec = elapsed_ms / 1000.0;
    double keys_per_sec = total_keys / elapsed_sec;
    double speed_power = std::log2(keys_per_sec);

    int host_found = 0;
    uint32_t host_found_key = 0;
    cudaMemcpyFromSymbol(&host_found, d_found, sizeof(int));

    if (host_found) {
        cudaMemcpyFromSymbol(&host_found_key, d_found_key, sizeof(uint32_t));
        std::cout << " Result        : Key = 0x" << std::hex << std::setw(8) << std::setfill('0') << std::uppercase << host_found_key << std::dec << "\n";
    }
    else {
        std::cout << " Result        : Key not found\n";
    }
    std::cout << " Time          : " << std::fixed << std::setprecision(6) << elapsed_sec << " sec\n";
    std::cout << " Speed         : 2^" << std::fixed << std::setprecision(2) << speed_power << " keys/sec\n";

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}