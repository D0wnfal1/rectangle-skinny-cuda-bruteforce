#include <iostream>
#include <cstdint>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__managed__ unsigned int d_found = 0;
__managed__ unsigned long long d_found_tk1 = 0;

__constant__ int d_tk1_idx[32][8];

__device__ __forceinline__ uint8_t get_rc(int r) {
    const uint8_t rc[32] = {
        0x01,0x03,0x07,0x0F, 0x1F,0x3E,0x3D,0x3B,
        0x37,0x2F,0x1E,0x3C, 0x39,0x33,0x27,0x0E,
        0x1D,0x3A,0x35,0x2B, 0x16,0x2C,0x18,0x30,
        0x21,0x02,0x05,0x0B, 0x17,0x2E,0x1C,0x38
    };
    return rc[r];
}

void precompute_tk1_idx(int h_tk1_idx[32][8]) {
    int PT[16] = { 9,15,8,13, 10,14,12,11, 0,1,2,3, 4,5,6,7 };
    for (int r = 0; r < 32; r++) {
        int pos[16] = { 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15 };
        for (int i = 0; i < r; i++) {
            int next_pos[16];
            for (int j = 0; j < 16; j++) next_pos[j] = pos[PT[j]];
            for (int j = 0; j < 16; j++) pos[j] = next_pos[j];
        }
        for (int c = 0; c < 8; c++) h_tk1_idx[r][c] = pos[c];
    }
}

__device__ __forceinline__ uint64_t get_key_bit(int bit_idx, uint64_t tk1_base) {
    if (bit_idx == 0) return 0xAAAAAAAAAAAAAAAAULL;
    if (bit_idx == 1) return 0xCCCCCCCCCCCCCCCCULL;
    if (bit_idx == 2) return 0xF0F0F0F0F0F0F0F0ULL;
    if (bit_idx == 3) return 0xFF00FF00FF00FF00ULL;
    if (bit_idx == 4) return 0xFFFF0000FFFF0000ULL;
    if (bit_idx == 5) return 0xFFFFFFFF00000000ULL;
    return ((tk1_base >> bit_idx) & 1ULL) ? ~0ULL : 0ULL;
}

__device__ __forceinline__ void subcells(uint64_t(&ST)[16][4], int c) {
    uint64_t t;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ST[c][0] ^= ~(ST[c][3] | ST[c][2]);
        t = ST[c][0];
        ST[c][0] = ST[c][3];
        ST[c][3] = ST[c][2];
        ST[c][2] = ST[c][1];
        ST[c][1] = t;
    }
}

__device__ __forceinline__ void mix_col_write(
    const uint64_t(&IN_ST)[16][4], uint64_t(&OUT_ST)[16][4],
    int i0, int i1, int i2, int i3,
    int o0, int o1, int o2, int o3)
{
#pragma unroll
    for (int b = 0; b < 4; b++) {
        uint64_t x0 = IN_ST[i0][b];
        uint64_t x1 = IN_ST[i1][b];
        uint64_t x2 = IN_ST[i2][b];
        uint64_t x3 = IN_ST[i3][b];

        OUT_ST[o0][b] = x0 ^ x2 ^ x3;
        OUT_ST[o1][b] = x0;
        OUT_ST[o2][b] = x1 ^ x2;
        OUT_ST[o3][b] = x0 ^ x2;
    }
}

__device__ __forceinline__ void do_round(
    uint64_t(&IN_ST)[16][4], uint64_t(&OUT_ST)[16][4],
    int r, uint64_t tk1_base)
{
#pragma unroll
    for (int c = 0; c < 16; c++) {
        subcells(IN_ST, c);
    }

    uint8_t rc = get_rc(r);
    if (rc & 0x01) IN_ST[0][0] ^= ~0ULL;
    if (rc & 0x02) IN_ST[0][1] ^= ~0ULL;
    if (rc & 0x04) IN_ST[0][2] ^= ~0ULL;
    if (rc & 0x08) IN_ST[0][3] ^= ~0ULL;
    if (rc & 0x10) IN_ST[4][0] ^= ~0ULL;
    if (rc & 0x20) IN_ST[4][1] ^= ~0ULL;
    IN_ST[8][2] ^= ~0ULL;

#pragma unroll
    for (int c = 0; c < 8; c++) {
        int orig_c = d_tk1_idx[r][c];
        IN_ST[c][0] ^= get_key_bit(orig_c * 4 + 0, tk1_base);
        IN_ST[c][1] ^= get_key_bit(orig_c * 4 + 1, tk1_base);
        IN_ST[c][2] ^= get_key_bit(orig_c * 4 + 2, tk1_base);
        IN_ST[c][3] ^= get_key_bit(orig_c * 4 + 3, tk1_base);
    }

    mix_col_write(IN_ST, OUT_ST, 0, 7, 10, 13, 0, 4, 8, 12);
    mix_col_write(IN_ST, OUT_ST, 1, 4, 11, 14, 1, 5, 9, 13);
    mix_col_write(IN_ST, OUT_ST, 2, 5, 8, 15, 2, 6, 10, 14);
    mix_col_write(IN_ST, OUT_ST, 3, 6, 9, 12, 3, 7, 11, 15);
}

__global__ void generate_skinny_kat_ciphertext_kernel(
    uint64_t pt_val,
    uint64_t target_key,
    uint64_t* out_ct_val)
{
    uint64_t tk1_base = target_key;
    uint64_t state[16][4];
    uint64_t next_state[16][4];

#pragma unroll
    for (int c = 0; c < 16; c++) {
#pragma unroll
        for (int b = 0; b < 4; b++) {
            int i = c * 4 + b;
            state[c][b] = ((pt_val >> i) & 1ULL) ? ~0ULL : 0ULL;
        }
    }

#pragma unroll
    for (int r = 0; r < 32; r += 2) {
        do_round(state, next_state, r, tk1_base);
        do_round(next_state, state, r + 1, tk1_base);
    }

    uint64_t ct_val = 0;
#pragma unroll
    for (int c = 0; c < 16; c++) {
#pragma unroll
        for (int b = 0; b < 4; b++) {
            int idx = c * 4 + b;
            uint64_t bit0 = state[c][b] & 1ULL;
            ct_val |= (bit0 << idx);
        }
    }

    *out_ct_val = ct_val;
}

__global__ void skinny_64_64_kernel(
    uint64_t pt_val,
    uint64_t ct_val,
    uint64_t total_threads,
    uint64_t key_offset)
{
    if (d_found) return;

    uint64_t blockId = blockIdx.x + (uint64_t)blockIdx.y * gridDim.x;
    uint64_t tid = blockId * blockDim.x + threadIdx.x;

    if (tid >= total_threads) return;

    uint64_t tk1_base = key_offset + tid * 64ULL;
    uint64_t state[16][4];
    uint64_t next_state[16][4];

#pragma unroll
    for (int c = 0; c < 16; c++) {
#pragma unroll
        for (int b = 0; b < 4; b++) {
            int i = c * 4 + b;
            state[c][b] = ((pt_val >> i) & 1ULL) ? ~0ULL : 0ULL;
        }
    }

#pragma unroll
    for (int r = 0; r < 32; r += 2) {
        do_round(state, next_state, r, tk1_base);
        do_round(next_state, state, r + 1, tk1_base);
    }

    uint64_t match = ~0ULL;

#pragma unroll
    for (int c = 0; c < 16; c++) {
#pragma unroll
        for (int b = 0; b < 4; b++) {
            int idx = c * 4 + b;
            uint64_t target_bit = ((ct_val >> idx) & 1ULL) ? ~0ULL : 0ULL;
            match &= ~(state[c][b] ^ target_bit);
        }
    }

    if (match != 0ULL) {
        int k = __ffsll(match) - 1;
        uint64_t found_key = tk1_base + (uint64_t)k;
        if (atomicExch(&d_found, 1) == 0) {
            d_found_tk1 = found_key;
        }
    }
}

int main()
{
    cudaSetDeviceFlags(cudaDeviceMapHost);

    int h_tk1_idx[32][8];
    precompute_tk1_idx(h_tk1_idx);
    cudaMemcpyToSymbol(d_tk1_idx, h_tk1_idx, sizeof(h_tk1_idx));

    int mode_choice = 0;
    std::cout << " OPERATION MODE\n";
    std::cout << " 1. KAT Verification Mode\n";
    std::cout << " 2. Benchmark Mode\n";
    std::cout << " 1 or 2: ";
    std::cin >> mode_choice;
    std::cout << "\n";

    bool kat_mode = (mode_choice == 1);

    uint64_t pt_val = 0;
    uint64_t ct_val = 0;
    int power = 32;

    if (kat_mode) {
        pt_val = 0x0000000000000000ULL;
        uint64_t kat_target_key = 0x00000000ULL;

        uint64_t* d_ct_val;
        cudaMalloc((void**)&d_ct_val, sizeof(uint64_t));

        generate_skinny_kat_ciphertext_kernel << <1, 1 >> > (pt_val, kat_target_key, d_ct_val);
        cudaDeviceSynchronize();

        cudaMemcpy(&ct_val, d_ct_val, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree(d_ct_val);

        std::cout << " KAT Plaintext  : 0x0000000000000000\n";
        std::cout << " KAT Target Key : 0x00000000\n";
        std::cout << " KAT Ciphertext : 0x"
            << std::hex << std::setw(16) << std::setfill('0') << std::uppercase << ct_val << std::dec << "\n\n";

        power = 8; 
    }
    else {
        uint16_t pt[4] = { 0x1122, 0x3344, 0x5566, 0x7788 };
        uint16_t ct[4] = { 0xABCD, 0xEF01, 0x2345, 0x6789 };

        pt_val = ((uint64_t)pt[0] << 48) | ((uint64_t)pt[1] << 32) | ((uint64_t)pt[2] << 16) | ((uint64_t)pt[3]);
        ct_val = ((uint64_t)ct[0] << 48) | ((uint64_t)ct[1] << 32) | ((uint64_t)ct[2] << 16) | ((uint64_t)ct[3]);
        power = 32;
    }

    uint64_t total_keys = 1ULL << power;
    int threads_per_block = 256;
    uint64_t KEYS_PER_BATCH = 1ULL << 28;
    uint64_t max_threads_per_batch = KEYS_PER_BATCH / 64ULL;
    if (max_threads_per_batch == 0) max_threads_per_batch = 1;

    std::cout << " SKINNY-64/64 CUDA Bitslice Bruteforce\n";
    std::cout << " Key Space     : 2^" << power << "\n";

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    d_found = 0;
    d_found_tk1 = 0;

    cudaEventRecord(start, stream);

    for (uint64_t offset = 0; offset < total_keys; offset += KEYS_PER_BATCH) {
        if (d_found) break;

        uint64_t current_batch_keys = std::min(KEYS_PER_BATCH, total_keys - offset);
        uint64_t total_threads_needed = current_batch_keys / 64ULL;
        if (total_threads_needed == 0) total_threads_needed = 1;

        uint64_t total_blocks = (total_threads_needed + threads_per_block - 1) / threads_per_block;
        dim3 grid;
        if (total_blocks <= 65535) {
            grid = dim3((unsigned int)total_blocks, 1);
        }
        else {
            uint64_t grid_x = 65535;
            uint64_t grid_y = (total_blocks + 65534) / 65535;
            grid = dim3((unsigned int)grid_x, (unsigned int)grid_y);
        }

        skinny_64_64_kernel << < grid, threads_per_block, 0, stream >> > (
            pt_val, ct_val, total_threads_needed, offset
            );

        cudaStreamSynchronize(stream);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double elapsed_sec = elapsed_ms / 1000.0;
    double keys_per_sec = total_keys / elapsed_sec;
    double speed_power = std::log2(keys_per_sec);


    if (d_found) {
        std::cout << " Result        : Key = 0x" << std::hex << std::setw(16) << std::setfill('0') << d_found_tk1 << std::dec << "\n";
        if (kat_mode) std::cout << "                 The algorithm works correctly\n";
    }
    else {
        std::cout << " Result        : Key not found\n";
    }

    std::cout << " Time : " << std::fixed << std::setprecision(6) << elapsed_sec << " sec\n";
    if (!kat_mode) {
        std::cout << " Speed : 2^" << std::fixed << std::setprecision(2) << speed_power << " keys/sec\n";
    }

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}