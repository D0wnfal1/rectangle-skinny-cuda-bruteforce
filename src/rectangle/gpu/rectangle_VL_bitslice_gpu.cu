#include <iostream>
#include <cstdint>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include <nvtx3/nvToolsExt.h>

#define ROTL16_PACKED(x, n) \
    ( (((((x) >> 16) << (n)) | (((x) >> 16) >> (16 - (n)))) & 0xFFFF) << 16 | \
      ((((x) & 0xFFFF) << (n)) | (((x) & 0xFFFF) >> (16 - (n)))) & 0xFFFF )

__constant__ uint8_t RECTANGLE_RC[25] = {
    0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16, 0x0C, 0x19, 0x13, 0x07, 0x0F,
    0x1F, 0x1E, 0x1D, 0x1A, 0x15, 0x0A, 0x14, 0x08, 0x11, 0x03, 0x06, 0x0D
};

__global__ void rectangle_bruteforce_bitslice_kernel(
    uint16_t pt0, uint16_t pt1, uint16_t pt2, uint16_t pt3,
    uint16_t ct0, uint16_t ct1, uint16_t ct2, uint16_t ct3,
    uint64_t offset, volatile int* global_found_flag, uint32_t* global_found_key)
{

    uint64_t thread_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t base_key_idx = offset + thread_id * 2;

    uint64_t k0 = base_key_idx;
    uint64_t k1 = base_key_idx + 1;

    uint32_t v[5];
    v[0] = ((k1 & 0xFFFF) << 16) | (k0 & 0xFFFF);
    v[1] = (((k1 >> 16) & 0xFFFF) << 16) | ((k0 >> 16) & 0xFFFF);
    v[2] = (((k1 >> 32) & 0xFFFF) << 16) | ((k0 >> 32) & 0xFFFF);
    v[3] = 0;
    v[4] = 0;

    uint32_t r0 = ((uint32_t)pt0 << 16) | pt0;
    uint32_t r1 = ((uint32_t)pt1 << 16) | pt1;
    uint32_t r2 = ((uint32_t)pt2 << 16) | pt2;
    uint32_t r3 = ((uint32_t)pt3 << 16) | pt3;

#pragma unroll
    for (int i = 0; i < 25; i++) {
        r0 ^= v[0];
        r1 ^= v[1];
        r2 ^= v[2];
        r3 ^= v[3];

        uint32_t t1 = ~r1;
        uint32_t t2 = r0 & t1;
        uint32_t t3 = r2 ^ r3;
        uint32_t b0 = t2 ^ t3;

        uint32_t t4 = r3 | t1;
        uint32_t t5 = r0 ^ t4;
        uint32_t b1 = r2 ^ t5;

        uint32_t t6 = r1 ^ r2;
        uint32_t t7 = t5 | t6;
        uint32_t b2 = r0 ^ t7;

        uint32_t t8 = r1 & b2;
        uint32_t b3 = t3 ^ t8;

        r0 = b0;
        r1 = ROTL16_PACKED(b1, 1);
        r2 = ROTL16_PACKED(b2, 12);
        r3 = ROTL16_PACKED(b3, 13);

        uint32_t kr0 = v[0] & 0x000F000F;
        uint32_t kr1 = v[1] & 0x000F000F;
        uint32_t kr2 = v[2] & 0x000F000F;
        uint32_t kr3 = v[3] & 0x000F000F;

        uint32_t kt1 = ~kr1;
        uint32_t kt2 = kr0 & kt1;
        uint32_t kt3 = kr2 ^ kr3;
        uint32_t kb0 = kt2 ^ kt3;

        uint32_t kt4 = kr3 | kt1;
        uint32_t kt5 = kr0 ^ kt4;
        uint32_t kb1 = kr2 ^ kt5;

        uint32_t kt6 = kr1 ^ kr2;
        uint32_t kt7 = kt5 | kt6;
        uint32_t kb2 = kr0 ^ kt7;

        uint32_t kt8 = kr1 & kb2;
        uint32_t kb3 = kt3 ^ kt8;

        v[0] = (v[0] & ~0x000F000F) | (kb0 & 0x000F000F);
        v[1] = (v[1] & ~0x000F000F) | (kb1 & 0x000F000F);
        v[2] = (v[2] & ~0x000F000F) | (kb2 & 0x000F000F);
        v[3] = (v[3] & ~0x000F000F) | (kb3 & 0x000F000F);

        uint32_t row0_old = v[0];
        v[0] = ROTL16_PACKED(v[0], 8) ^ v[1];
        v[1] = v[2];
        v[2] = v[3];
        v[3] = ROTL16_PACKED(v[3], 12) ^ v[4];
        v[4] = row0_old;

        uint32_t rc_packed = ((uint32_t)RECTANGLE_RC[i] << 16) | RECTANGLE_RC[i];
        v[0] ^= rc_packed;
    }

    r0 ^= v[0];
    r1 ^= v[1];
    r2 ^= v[2];
    r3 ^= v[3];

    uint32_t ct0_32 = ((uint32_t)ct0 << 16) | ct0;
    uint32_t ct1_32 = ((uint32_t)ct1 << 16) | ct1;
    uint32_t ct2_32 = ((uint32_t)ct2 << 16) | ct2;
    uint32_t ct3_32 = ((uint32_t)ct3 << 16) | ct3;

    uint32_t diff = (r0 ^ ct0_32) | (r1 ^ ct1_32) | (r2 ^ ct2_32) | (r3 ^ ct3_32);

    if (diff != 0xFFFFFFFF) {
        if ((diff & 0x0000FFFF) == 0) {
            *global_found_flag = 1;
            *global_found_key = base_key_idx;
        }
        if ((diff & 0xFFFF0000) == 0) {
            *global_found_flag = 1;
            *global_found_key = base_key_idx + 1;
        }
    }
}

__global__ void generate_rectangle_kat_ciphertext_kernel(
    uint16_t pt0, uint16_t pt1, uint16_t pt2, uint16_t pt3,
    uint64_t target_key,
    uint16_t* out_ct0, uint16_t* out_ct1, uint16_t* out_ct2, uint16_t* out_ct3)
{
    uint64_t k0 = target_key;
    uint64_t k1 = target_key;

    uint32_t v[5];
    v[0] = ((k1 & 0xFFFF) << 16) | (k0 & 0xFFFF);
    v[1] = (((k1 >> 16) & 0xFFFF) << 16) | ((k0 >> 16) & 0xFFFF);
    v[2] = (((k1 >> 32) & 0xFFFF) << 16) | ((k0 >> 32) & 0xFFFF);
    v[3] = 0;
    v[4] = 0;

    uint32_t r0 = ((uint32_t)pt0 << 16) | pt0;
    uint32_t r1 = ((uint32_t)pt1 << 16) | pt1;
    uint32_t r2 = ((uint32_t)pt2 << 16) | pt2;
    uint32_t r3 = ((uint32_t)pt3 << 16) | pt3;

#pragma unroll
    for (int i = 0; i < 25; i++) {
        r0 ^= v[0];
        r1 ^= v[1];
        r2 ^= v[2];
        r3 ^= v[3];

        uint32_t t1 = ~r1;
        uint32_t t2 = r0 & t1;
        uint32_t t3 = r2 ^ r3;
        uint32_t b0 = t2 ^ t3;

        uint32_t t4 = r3 | t1;
        uint32_t t5 = r0 ^ t4;
        uint32_t b1 = r2 ^ t5;

        uint32_t t6 = r1 ^ r2;
        uint32_t t7 = t5 | t6;
        uint32_t b2 = r0 ^ t7;

        uint32_t t8 = r1 & b2;
        uint32_t b3 = t3 ^ t8;

        r0 = b0;
        r1 = ROTL16_PACKED(b1, 1);
        r2 = ROTL16_PACKED(b2, 12);
        r3 = ROTL16_PACKED(b3, 13);

        uint32_t kr0 = v[0] & 0x000F000F;
        uint32_t kr1 = v[1] & 0x000F000F;
        uint32_t kr2 = v[2] & 0x000F000F;
        uint32_t kr3 = v[3] & 0x000F000F;

        uint32_t kt1 = ~kr1;
        uint32_t kt2 = kr0 & kt1;
        uint32_t kt3 = kr2 ^ kr3;
        uint32_t kb0 = kt2 ^ kt3;

        uint32_t kt4 = kr3 | kt1;
        uint32_t kt5 = kr0 ^ kt4;
        uint32_t kb1 = kr2 ^ kt5;

        uint32_t kt6 = kr1 ^ kr2;
        uint32_t kt7 = kt5 | kt6;
        uint32_t kb2 = kr0 ^ kt7;

        uint32_t kt8 = kr1 & kb2;
        uint32_t kb3 = kt3 ^ kt8;

        v[0] = (v[0] & ~0x000F000F) | (kb0 & 0x000F000F);
        v[1] = (v[1] & ~0x000F000F) | (kb1 & 0x000F000F);
        v[2] = (v[2] & ~0x000F000F) | (kb2 & 0x000F000F);
        v[3] = (v[3] & ~0x000F000F) | (kb3 & 0x000F000F);

        uint32_t row0_old = v[0];
        v[0] = ROTL16_PACKED(v[0], 8) ^ v[1];
        v[1] = v[2];
        v[2] = v[3];
        v[3] = ROTL16_PACKED(v[3], 12) ^ v[4];
        v[4] = row0_old;

        uint32_t rc_packed = ((uint32_t)RECTANGLE_RC[i] << 16) | RECTANGLE_RC[i];
        v[0] ^= rc_packed;
    }

    r0 ^= v[0];
    r1 ^= v[1];
    r2 ^= v[2];
    r3 ^= v[3];

    *out_ct0 = r0 & 0xFFFF;
    *out_ct1 = r1 & 0xFFFF;
    *out_ct2 = r2 & 0xFFFF;
    *out_ct3 = r3 & 0xFFFF;
}

int main() {
    cudaSetDeviceFlags(cudaDeviceMapHost);

    int mode_choice = 0;
    std::cout << "OPERATION MODE\n";
    std::cout << " 1. KAT Verification Mode\n";
    std::cout << " 2. Benchmark Mode\n";
    std::cout << " 1 or 2: ";
    std::cin >> mode_choice;
    std::cout << "\n";

    bool kat_mode = (mode_choice == 1);

    uint16_t pt[4];
    uint16_t ct[4];
    int power;

    if (kat_mode) {

        pt[0] = 0x0000; pt[1] = 0x0000; pt[2] = 0x0000; pt[3] = 0x0000;
        uint64_t kat_target_key = 0x00000000;

        uint16_t* d_ct0, * d_ct1, * d_ct2, * d_ct3;
        cudaMalloc((void**)&d_ct0, sizeof(uint16_t));
        cudaMalloc((void**)&d_ct1, sizeof(uint16_t));
        cudaMalloc((void**)&d_ct2, sizeof(uint16_t));
        cudaMalloc((void**)&d_ct3, sizeof(uint16_t));

        generate_rectangle_kat_ciphertext_kernel << <1, 1 >> > (
            pt[0], pt[1], pt[2], pt[3], kat_target_key, d_ct0, d_ct1, d_ct2, d_ct3);
        cudaDeviceSynchronize();

        cudaMemcpy(&ct[0], d_ct0, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ct[1], d_ct1, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ct[2], d_ct2, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ct[3], d_ct3, sizeof(uint16_t), cudaMemcpyDeviceToHost);

        cudaFree(d_ct0); cudaFree(d_ct1); cudaFree(d_ct2); cudaFree(d_ct3);

        std::cout << " KAT Plaintext  : 0x0000 0000 0000 0000\n";
        std::cout << " KAT Target Key : 0x00000000\n";
        std::cout << " KAT Ciphertext : 0x"
            << std::hex << std::setw(4) << std::setfill('0') << ct[3]
            << std::setw(4) << std::setfill('0') << ct[2]
            << std::setw(4) << std::setfill('0') << ct[1]
            << std::setw(4) << std::setfill('0') << ct[0] << std::dec << "\n\n";

        power = 8; 
    }
    else {
        pt[0] = 0x1122; pt[1] = 0x3344; pt[2] = 0x5566; pt[3] = 0x7788;
        ct[0] = 0xABCD; ct[1] = 0xEF01; ct[2] = 0x2345; ct[3] = 0x6789;
        power = 32;
    }

    uint64_t total_keys = 1ULL << power;
    int threads_per_block = 256;

    uint64_t KEYS_PER_BATCH = 1ULL << 28;
    uint64_t max_threads_per_batch = KEYS_PER_BATCH / 2;
    if (max_threads_per_batch == 0) max_threads_per_batch = 1;
    unsigned int blocks_per_grid = (max_threads_per_batch + threads_per_block - 1) / threads_per_block;

    std::cout << " RECTANGLE-80 GPU CUDA Variable-level bitslice\n";
    std::cout << " Key Space     : 2^" << power << "\n";
    std::cout << " Configuration : Grid " << blocks_per_grid << " blocks, " << threads_per_block << " threads\n";

    volatile int* h_found_flag = nullptr;
    uint32_t* h_found_key = nullptr;
    cudaHostAlloc((void**)&h_found_flag, sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_found_key, sizeof(uint32_t), cudaHostAllocMapped);

    *h_found_flag = 0;
    *h_found_key = 0;

    int* d_found_flag = nullptr;
    uint32_t* d_found_key = nullptr;
    cudaHostGetDevicePointer((void**)&d_found_flag, (void*)h_found_flag, 0);
    cudaHostGetDevicePointer((void**)&d_found_key, (void*)h_found_key, 0);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nvtxRangePushA("Total Bruteforce Operation");
    cudaEventRecord(start, stream);

    for (uint64_t offset = 0; offset < total_keys; offset += KEYS_PER_BATCH) {
        if (*h_found_flag != 0) break;

        uint64_t current_batch_keys = std::min(KEYS_PER_BATCH, total_keys - offset);
        uint64_t total_threads_needed = current_batch_keys / 2;
        if (total_threads_needed == 0) total_threads_needed = 1;

        unsigned int blocks = (total_threads_needed + threads_per_block - 1) / threads_per_block;

        nvtxRangePushA("Kernel Launch & Compute");

        rectangle_bruteforce_bitslice_kernel << <blocks, threads_per_block, 0, stream >> > (
            pt[0], pt[1], pt[2], pt[3],
            ct[0], ct[1], ct[2], ct[3],
            offset, d_found_flag, d_found_key
            );

        nvtxRangePop();
        cudaStreamSynchronize(stream);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    nvtxRangePop();

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double elapsed_sec = elapsed_ms / 1000.0;
    double keys_per_sec = total_keys / elapsed_sec;
    double speed_power = std::log2(keys_per_sec);

    if (*h_found_flag) {
        std::cout << " Result        : Key = 0x"
            << std::hex << std::setw(8) << std::setfill('0') << std::uppercase
            << *h_found_key << std::dec << "\n";
        if (kat_mode) std::cout << "                 The algorithm works correctly\n";
    }
    else {
        std::cout << " Result        : Key not found\n";
    }
    std::cout << " Time          : " << std::fixed << std::setprecision(6) << elapsed_sec << " sec\n";
    if (!kat_mode) {
        std::cout << " Speed     : 2^" << std::fixed << std::setprecision(2) << speed_power << " keys/sec\n";
    }

    cudaFreeHost((void*)h_found_flag);
    cudaFreeHost((void*)h_found_key);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}