#include <iostream>
#include <cstdint>
#include <iomanip>
#include <cmath>

#define RECTANGLE_ROTL16(x, n) (uint16_t)(((x) << (n)) | ((x) >> (16 - (n))))

__device__ int d_found = 0;
__device__ uint32_t d_found_key = 0;

__constant__ uint8_t RECTANGLE_RC[25] = {
    0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16, 0x0C, 0x19, 0x13, 0x07, 0x0F,
    0x1F, 0x1E, 0x1D, 0x1A, 0x15, 0x0A, 0x14, 0x08, 0x11, 0x03, 0x06, 0x0D
};

__forceinline__ __device__ void rectangle_key_schedule_bitslice(uint16_t master_key[5], uint16_t round_keys[26][4]) {
    uint16_t v[5] = { master_key[0], master_key[1], master_key[2], master_key[3], master_key[4] };

#pragma unroll
    for (int i = 0; i < 26; i++) {
        round_keys[i][0] = v[0];
        round_keys[i][1] = v[1];
        round_keys[i][2] = v[2];
        round_keys[i][3] = v[3];

        if (i == 25) break;

        uint16_t r0 = v[0] & 0x0F;
        uint16_t r1 = v[1] & 0x0F;
        uint16_t r2 = v[2] & 0x0F;
        uint16_t r3 = v[3] & 0x0F;

        uint16_t t1 = ~r1;
        uint16_t t2 = r0 & t1;
        uint16_t t3 = r2 ^ r3;
        uint16_t b0 = t2 ^ t3;

        uint16_t t4 = r3 | t1;
        uint16_t t5 = r0 ^ t4;
        uint16_t b1 = r2 ^ t5;

        uint16_t t6 = r1 ^ r2;
        uint16_t t7 = t5 | t6;
        uint16_t b2 = r0 ^ t7;

        uint16_t t8 = r1 & b2;
        uint16_t b3 = t3 ^ t8;

        v[0] = (v[0] & ~0x0F) | (b0 & 0x0F);
        v[1] = (v[1] & ~0x0F) | (b1 & 0x0F);
        v[2] = (v[2] & ~0x0F) | (b2 & 0x0F);
        v[3] = (v[3] & ~0x0F) | (b3 & 0x0F);

        uint16_t row0_old = v[0];
        v[0] = RECTANGLE_ROTL16(v[0], 8) ^ v[1];
        v[1] = v[2];
        v[2] = v[3];
        v[3] = RECTANGLE_ROTL16(v[3], 12) ^ v[4];
        v[4] = row0_old;

        v[0] ^= RECTANGLE_RC[i];
    }
}

__forceinline__ __device__ void rectangle_encrypt_bitslice(uint16_t state[4], const uint16_t round_keys[26][4]) {
    uint16_t r0 = state[0], r1 = state[1], r2 = state[2], r3 = state[3];

#pragma unroll
    for (int rnd = 0; rnd < 25; ++rnd) {
        r0 ^= round_keys[rnd][0];
        r1 ^= round_keys[rnd][1];
        r2 ^= round_keys[rnd][2];
        r3 ^= round_keys[rnd][3];

        uint16_t t1 = ~r1;
        uint16_t t2 = r0 & t1;
        uint16_t t3 = r2 ^ r3;
        uint16_t b0 = t2 ^ t3;

        uint16_t t4 = r3 | t1;
        uint16_t t5 = r0 ^ t4;
        uint16_t b1 = r2 ^ t5;

        uint16_t t6 = r1 ^ r2;
        uint16_t t7 = t5 | t6;
        uint16_t b2 = r0 ^ t7;

        uint16_t t8 = r1 & b2;
        uint16_t b3 = t3 ^ t8;

        r0 = b0;
        r1 = b1;
        r2 = b2;
        r3 = b3;

        r1 = RECTANGLE_ROTL16(r1, 1);
        r2 = RECTANGLE_ROTL16(r2, 12);
        r3 = RECTANGLE_ROTL16(r3, 13);
    }

    state[0] = r0 ^ round_keys[25][0];
    state[1] = r1 ^ round_keys[25][1];
    state[2] = r2 ^ round_keys[25][2];
    state[3] = r3 ^ round_keys[25][3];
}

__global__ void rectangle_bruteforce_bitslice_kernel(uint16_t pt0, uint16_t pt1, uint16_t pt2, uint16_t pt3,
    uint16_t ct0, uint16_t ct1, uint16_t ct2, uint16_t ct3) {
    if (d_found) return;

    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    uint16_t master_key[5] = { (uint16_t)(thread_id & 0xFFFF), (uint16_t)((thread_id >> 16) & 0xFFFF), 0, 0, 0 };
    uint16_t round_keys[26][4];

    rectangle_key_schedule_bitslice(master_key, round_keys);

    uint16_t state[4] = { pt0, pt1, pt2, pt3 };
    rectangle_encrypt_bitslice(state, round_keys);

    if (state[0] == ct0 && state[1] == ct1 && state[2] == ct2 && state[3] == ct3) {
        if (atomicExch(&d_found, 1) == 0) {
            d_found_key = thread_id;
        }
    }
}

int main() {
    uint16_t pt[4] = { 0x1122, 0x3344, 0x5566, 0x7788 };
    uint16_t ct[4] = { 0xABCD, 0xEF01, 0x2345, 0x6789 }; 

    int power = 32;

    uint64_t total_keys = 1ULL << power;
    int threads_per_block = 256;
    unsigned int blocks_per_grid = (unsigned int)(total_keys / threads_per_block);
    if (blocks_per_grid == 0) blocks_per_grid = 1;

    std::cout << " RECTANGLE-80 GPU CUDA Pure ALU Bit-Slice\n";
    std::cout << " Key Space     : 2^" << power << "\n";
    std::cout << " Configuration : Grid " << blocks_per_grid << " blocks, " << threads_per_block << " threads\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    rectangle_bruteforce_bitslice_kernel << <blocks_per_grid, threads_per_block >> > (
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
        std::cout << " Result        : Key = 0x" << std::hex << std::setw(8) << std::setfill('0') << std::uppercase << host_found_key << std::dec << "\n";
    }
    else {
        std::cout << " Result        : Key not found\n";
    }
    std::cout << " Time          : " << std::fixed << std::setprecision(6) << elapsed_sec << " sec\n";
    std::cout << " Speed         : 2^" << std::fixed << std::setprecision(2) << speed_power << " keys/sec\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}