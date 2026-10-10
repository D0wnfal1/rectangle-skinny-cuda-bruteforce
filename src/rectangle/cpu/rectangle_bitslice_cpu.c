#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>

#define RECTANGLE_ROTL16(x, n) (uint16_t)(((x) << (n)) | ((x) >> (16 - (n))))
#define RECTANGLE_ROTL32(x, n) (uint32_t)(((x) << (n)) | ((x) >> (32 - (n))))

static const uint8_t RECTANGLE_RC[25] = {
    0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16, 0x0C, 0x19, 0x13, 0x07, 0x0F,
    0x1F, 0x1E, 0x1D, 0x1A, 0x15, 0x0A, 0x14, 0x08, 0x11, 0x03, 0x06, 0x0D
};

void rectangle_key_schedule_bitslice(uint16_t master_key[5], uint16_t round_keys[26][4]) {
    uint16_t v[5] = { master_key[0], master_key[1], master_key[2], master_key[3], master_key[4] };

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

static inline void rectangle_encrypt_bitslice(uint16_t state[4], const uint16_t round_keys[26][4]) {
    uint16_t r0 = state[0], r1 = state[1], r2 = state[2], r3 = state[3];

    for (int rnd = 0; rnd < 25; ++rnd) {
        r0 ^= round_keys[rnd][0]; r1 ^= round_keys[rnd][1];
        r2 ^= round_keys[rnd][2]; r3 ^= round_keys[rnd][3];

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

        r0 = b0; r1 = b1; r2 = b2; r3 = b3;
        r1 = RECTANGLE_ROTL16(r1, 1);
        r2 = RECTANGLE_ROTL16(r2, 12);
        r3 = RECTANGLE_ROTL16(r3, 13);
    }
    state[0] = r0 ^ round_keys[25][0];
    state[1] = r1 ^ round_keys[25][1];
    state[2] = r2 ^ round_keys[25][2];
    state[3] = r3 ^ round_keys[25][3];
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    int power = 16;

    uint64_t total_keys = 1ULL << power;
    uint32_t dummy_sink = 0;
    int num_threads = omp_get_max_threads();

    printf(" RECTANGLE-80 CPU OpenMP Bitslice\n");
    printf(" Key Space     : 2^%d \n", power);
    printf(" Configuration : %d threads\n", num_threads);

    LARGE_INTEGER frequency, t1, t2;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);

    int64_t signed_total_keys = (int64_t)total_keys;
    int64_t iter;

#pragma omp parallel for reduction(^:dummy_sink) schedule(static)
    for (iter = 0; iter < signed_total_keys; ++iter) {
        uint16_t state[4] = { 0, 0, 0, 0 };
        uint16_t round_keys[26][4] = { 0 };

        uint16_t master_key[5] = { (uint16_t)(iter & 0xFFFF), (uint16_t)((iter >> 16) & 0xFFFF), 0, 0, 0 };
        rectangle_key_schedule_bitslice(master_key, round_keys);

        rectangle_encrypt_bitslice(state, round_keys);
        dummy_sink ^= state[0];
    }

    QueryPerformanceCounter(&t2);

    double elapsed_sec = (double)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
    double keys_per_sec = (double)total_keys / elapsed_sec;
    double speed_power = log2(keys_per_sec); 

    printf(" Time          : %.6f sec\n", elapsed_sec);
    printf(" Speed         : 2^%.2f keys/sec\n", speed_power);

    return 0;
}