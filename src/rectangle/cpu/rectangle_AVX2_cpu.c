#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include <immintrin.h>

#if defined(_MSC_VER)
#define ALIGN32 __declspec(align(32))
#else
#define ALIGN32 __attribute__((aligned(32)))
#endif

#define RECTANGLE_ROTL16_AVX2(x, n) \
    _mm256_or_si256(_mm256_slli_epi16((x), (n)), _mm256_srli_epi16((x), 16 - (n)))

static const uint8_t RECTANGLE_RC[25] = {
    0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16, 0x0C, 0x19, 0x13, 0x07, 0x0F,
    0x1F, 0x1E, 0x1D, 0x1A, 0x15, 0x0A, 0x14, 0x08, 0x11, 0x03, 0x06, 0x0D
};

static inline void sbox_avx2(__m256i* r0, __m256i* r1, __m256i* r2, __m256i* r3) {
    __m256i t1 = _mm256_xor_si256(*r1, _mm256_set1_epi16(-1));
    __m256i t2 = _mm256_andnot_si256(*r1, *r0);                
    __m256i t3 = _mm256_xor_si256(*r2, *r3);
    __m256i b0 = _mm256_xor_si256(t2, t3);

    __m256i t4 = _mm256_or_si256(*r3, t1);
    __m256i t5 = _mm256_xor_si256(*r0, t4);
    __m256i b1 = _mm256_xor_si256(*r2, t5);

    __m256i t6 = _mm256_xor_si256(*r1, *r2);
    __m256i t7 = _mm256_or_si256(t5, t6);
    __m256i b2 = _mm256_xor_si256(*r0, t7);

    __m256i t8 = _mm256_and_si256(*r1, b2);
    __m256i b3 = _mm256_xor_si256(t3, t8);

    *r0 = b0; *r1 = b1; *r2 = b2; *r3 = b3;
}

void rectangle_key_schedule_avx2(__m256i master_key[5], __m256i round_keys[26][4]) {
    __m256i v[5] = { master_key[0], master_key[1], master_key[2], master_key[3], master_key[4] };
    __m256i mask0F = _mm256_set1_epi16(0x0F);
    __m256i maskF0 = _mm256_set1_epi16(~0x0F);

    for (int i = 0; i < 26; i++) {
        round_keys[i][0] = v[0];
        round_keys[i][1] = v[1];
        round_keys[i][2] = v[2];
        round_keys[i][3] = v[3];
        if (i == 25) break;

        __m256i r0 = _mm256_and_si256(v[0], mask0F);
        __m256i r1 = _mm256_and_si256(v[1], mask0F);
        __m256i r2 = _mm256_and_si256(v[2], mask0F);
        __m256i r3 = _mm256_and_si256(v[3], mask0F);

        sbox_avx2(&r0, &r1, &r2, &r3);

        v[0] = _mm256_or_si256(_mm256_and_si256(v[0], maskF0), _mm256_and_si256(r0, mask0F));
        v[1] = _mm256_or_si256(_mm256_and_si256(v[1], maskF0), _mm256_and_si256(r1, mask0F));
        v[2] = _mm256_or_si256(_mm256_and_si256(v[2], maskF0), _mm256_and_si256(r2, mask0F));
        v[3] = _mm256_or_si256(_mm256_and_si256(v[3], maskF0), _mm256_and_si256(r3, mask0F));

        __m256i row0_old = v[0];
        v[0] = _mm256_xor_si256(RECTANGLE_ROTL16_AVX2(v[0], 8), v[1]);
        v[1] = v[2];
        v[2] = v[3];
        v[3] = _mm256_xor_si256(RECTANGLE_ROTL16_AVX2(v[3], 12), v[4]);
        v[4] = row0_old;

        v[0] = _mm256_xor_si256(v[0], _mm256_set1_epi16(RECTANGLE_RC[i]));
    }
}

static inline void rectangle_encrypt_avx2(__m256i state[4], const __m256i round_keys[26][4]) {
    __m256i r0 = state[0], r1 = state[1], r2 = state[2], r3 = state[3];

    for (int rnd = 0; rnd < 25; ++rnd) {
        r0 = _mm256_xor_si256(r0, round_keys[rnd][0]);
        r1 = _mm256_xor_si256(r1, round_keys[rnd][1]);
        r2 = _mm256_xor_si256(r2, round_keys[rnd][2]);
        r3 = _mm256_xor_si256(r3, round_keys[rnd][3]);

        sbox_avx2(&r0, &r1, &r2, &r3);

        r1 = RECTANGLE_ROTL16_AVX2(r1, 1);
        r2 = RECTANGLE_ROTL16_AVX2(r2, 12);
        r3 = RECTANGLE_ROTL16_AVX2(r3, 13);
    }

    state[0] = _mm256_xor_si256(r0, round_keys[25][0]);
    state[1] = _mm256_xor_si256(r1, round_keys[25][1]);
    state[2] = _mm256_xor_si256(r2, round_keys[25][2]);
    state[3] = _mm256_xor_si256(r3, round_keys[25][3]);
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    int power = 32;

    uint64_t total_keys = 1ULL << power;
    uint32_t dummy_sink = 0;
    int num_threads = omp_get_max_threads();

    printf(" RECTANGLE-80 CPU AVX2 OpenMP Bitslice\n");
    printf(" Key Space     : 2^%d\n", power);
    printf(" Configuration : %d threads\n", num_threads);

    LARGE_INTEGER frequency, t1, t2;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);

    int64_t signed_total_keys = (int64_t)total_keys;
    int64_t iter;

    const __m256i offsets = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

#pragma omp parallel for reduction(^:dummy_sink) schedule(static)
    for (iter = 0; iter < signed_total_keys; iter += 16) {
        __m256i state[4] = {
            _mm256_setzero_si256(), _mm256_setzero_si256(),
            _mm256_setzero_si256(), _mm256_setzero_si256()
        };
        __m256i round_keys[26][4];
        __m256i master_key[5];

        uint16_t base0 = (uint16_t)(iter & 0xFFFF);
        master_key[0] = _mm256_add_epi16(_mm256_set1_epi16(base0), offsets);

        uint16_t base1 = (uint16_t)((iter >> 16) & 0xFFFF);
        master_key[1] = _mm256_set1_epi16(base1);

        master_key[2] = _mm256_setzero_si256();
        master_key[3] = _mm256_setzero_si256();
        master_key[4] = _mm256_setzero_si256();

        rectangle_key_schedule_avx2(master_key, round_keys);
        rectangle_encrypt_avx2(state, round_keys);

        ALIGN32 uint16_t res[16];
        _mm256_store_si256((__m256i*)res, state[0]);

        int limit = (int)((signed_total_keys - iter < 16) ? (signed_total_keys - iter) : 16);
        for (int j = 0; j < limit; j++) {
            dummy_sink ^= res[j];
        }
    }

    QueryPerformanceCounter(&t2);

    double elapsed_sec = (double)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
    double keys_per_sec = (double)total_keys / elapsed_sec;
    double speed_power = log2(keys_per_sec);

    printf(" Time          : %.6f sec\n", elapsed_sec);
    printf(" Speed         : 2^%.2f keys/sec\n", speed_power);

    return 0;
}