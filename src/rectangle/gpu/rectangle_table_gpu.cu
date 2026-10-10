#include <iostream>
#include <cstdint>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "tables_RECTANGLE.inc"
#include <cmath>

#define RECTANGLE_ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

__device__ int d_found = 0;
__device__ uint32_t d_found_key = 0;

__device__ uint64_t pack_column_major(uint16_t r0, uint16_t r1, uint16_t r2, uint16_t r3) {
    uint64_t state = 0;
    for (int i = 0; i < 16; i++) {
        uint64_t col = ((r3 >> i) & 1) << 3 | ((r2 >> i) & 1) << 2 |
            ((r1 >> i) & 1) << 1 | ((r0 >> i) & 1);
        state |= (col << (4 * i));
    }
    return state;
}

__global__ void rectangle_bruteforce_optimized_kernel(uint64_t pt, uint64_t ct, unsigned int exhaustive_loops) {
    if (d_found) return;

    __shared__ uint64_t s_T[256];
    if (threadIdx.x < 256) {
        s_T[threadIdx.x] = T_TABLE[threadIdx.x];
    }
    __syncthreads();

    const uint8_t sbox[16] = { 0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9, 0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2 };
    const uint8_t rc[25] = { 0x01,0x02,0x04,0x09,0x12,0x05,0x0B,0x16,0x0C,0x19,0x13,0x07,0x0F,0x1F,0x1E,0x1D,0x1A,0x15,0x0A,0x14,0x08,0x11,0x03,0x06,0x0D };

    uint32_t base_tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (unsigned int j = 0; j < exhaustive_loops; j++) {
        uint32_t tid = base_tid + j * stride;

        uint16_t v[5] = { (uint16_t)(tid & 0xFFFF), (uint16_t)((tid >> 16) & 0xFFFF), 0, 0, 0 };
        uint64_t state = pt;

        for (int rnd = 0; rnd < 25; ++rnd) {
            state ^= pack_column_major(v[0], v[1], v[2], v[3]);

            uint64_t next = 0;
            next ^= s_T[(state) & 0xFF];
            next ^= RECTANGLE_ROTL64(s_T[(state >> 8) & 0xFF], 8);
            next ^= RECTANGLE_ROTL64(s_T[(state >> 16) & 0xFF], 16);
            next ^= RECTANGLE_ROTL64(s_T[(state >> 24) & 0xFF], 24);
            next ^= RECTANGLE_ROTL64(s_T[(state >> 32) & 0xFF], 32);
            next ^= RECTANGLE_ROTL64(s_T[(state >> 40) & 0xFF], 40);
            next ^= RECTANGLE_ROTL64(s_T[(state >> 48) & 0xFF], 48);
            next ^= RECTANGLE_ROTL64(s_T[(state >> 56) & 0xFF], 56);
            state = next;

            for (int col = 0; col < 4; col++) {
                uint8_t nibble = ((v[3] >> col) & 1) << 3 | ((v[2] >> col) & 1) << 2 |
                    ((v[1] >> col) & 1) << 1 | ((v[0] >> col) & 1);
                uint8_t s = sbox[nibble];
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

            v[0] ^= rc[rnd];
        }
        state ^= pack_column_major(v[0], v[1], v[2], v[3]);

        if (state == ct) {
            if (atomicExch(&d_found, 1) == 0) {
                d_found_key = tid;
            }
        }
    }
}

int main() {
    uint64_t pt = 0x1122334455667788ULL;
    uint64_t ct = 0xABCDEF0123456789ULL;

    int power = 32;

    uint64_t total_keys = 1ULL << power;

    unsigned int threads_per_block = 256;
    unsigned int blocks_per_grid = 1;
    unsigned int exhaustive_loops = 1;

    if (total_keys > threads_per_block) {
        unsigned long long grids = total_keys / threads_per_block;
        if (grids > 65536) {
            blocks_per_grid = 65536;
            exhaustive_loops = (unsigned int)(grids / blocks_per_grid);
        }
        else {
            blocks_per_grid = (unsigned int)grids;
            exhaustive_loops = 1;
        }
    }

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    std::cout << " [RECTANGLE-80] GPU CUDA Table-based\n";
    std::cout << " Key Space     : 2^" << power << "\n";
    std::cout << " Configuration : Grid " << blocks_per_grid << " blocks, " << threads_per_block << " threads " << "\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    rectangle_bruteforce_optimized_kernel << <blocks_per_grid, threads_per_block >> > (pt, ct, exhaustive_loops);

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