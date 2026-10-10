# GPU Brute-Force Benchmarks for RECTANGLE and SKINNY (CUDA & AVX2)

Massively parallel exhaustive key search for the lightweight block ciphers **RECTANGLE-80** and **SKINNY-64/64**, implemented in several CPU and GPU variants (naive, table-based, bitsliced) and benchmarked on a single laptop GPU.

The code accompanies my B.Sc. thesis *"CUDA-Based Hardware–Software System for Massively Parallel Computing to Analyze the Resistance of the RECTANGLE and SKINNY Block Ciphers to Brute-Force Attacks"* (Computer Engineering, National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute", 2026). The thesis is written in Ukrainian; this README summarizes the method and results in English.

> **Scope.** This is a research and benchmarking prototype. Benchmarks scan small key ranges (typically 2^32 keys) to measure throughput; full 2^64 / 2^80 searches are only *extrapolated* from the measured numbers (see [Security margin](#estimated-security-margin)).

## Highlights

- Peak throughput on one NVIDIA GTX 1650 Mobile: **2^30.94 keys/s** for RECTANGLE-80 (≈ 2.1 × 10⁹) and **2^32.03 keys/s** for SKINNY-64/64 (≈ 4.4 × 10⁹).
- The bitsliced RECTANGLE kernel is **7.4× faster** than the table-based GPU version and ≈ 5.3× faster than the AVX2 + OpenMP CPU version. For SKINNY, bitslicing is ≈ 15.7× faster than the table-based kernel.
- Round functions and on-the-fly key generation run **entirely in registers**: no S-box tables, no shared/global-memory accesses inside the rounds, and no data-dependent branches.
- A straightforward ("naive") GPU port is *slower* than the vectorized CPU baseline, which shows that moving to a GPU only pays off with an algorithm–architecture co-design.

## Results

**Test bench:** NVIDIA GeForce GTX 1650 Mobile (Turing TU117, 1024 CUDA cores, 4 GB GDDR6), Intel Core i5-10300H host, Windows 64-bit, CUDA Toolkit 13.1, `nvcc -O3` with a per-thread register limit (`-maxrregcount`). Kernel time is measured with CUDA Events.

### RECTANGLE-80

| Variant | Platform | Throughput (keys/s) | Source file |
|---|---|---|---|
| Bitslice + OpenMP | CPU | 2^24.53 | `rectangle_bitslice_cpu.c` |
| Bitslice + AVX2 + OpenMP | CPU | 2^28.53 | `rectangle_AVX2_cpu.c` |
| Naive port | GPU (CUDA) | 2^26.14 | `rectangle_naive_gpu.cu` |
| Table-based | GPU (CUDA) | 2^28.05 | `rectangle_table_gpu.cu` |
| Bitslice ("Pure ALU") | GPU (CUDA) | 2^30.52 | `rectangle_bitslice_gpu.cu` |
| **Variable-level bitslice** | GPU (CUDA) | **2^30.94** | `rectangle_VL_bitslice_gpu.cu` |

### SKINNY-64/64

| Variant | Platform | Throughput (keys/s) | Source file |
|---|---|---|---|
| Table-based | GPU (CUDA) | 2^28.06 | `skinny_table_gpu.cu` |
| **64-way bitslice** | GPU (CUDA) | **2^32.03** | `skinny_VL_bitslice_gpu.cu` |

## Implementation notes

- **RECTANGLE, variable-level bitslicing.** The 64-bit state is held in four 16-bit "slices"; two independent states are packed into one 32-bit register, so each thread tests two keys at once. The S-box and linear layer are expressed as pure logic/rotate operations.
- **SKINNY, 64-way bitslicing.** Each thread processes 64 blocks in parallel using 64-bit registers. The tweakey permutation is resolved on the host into an index table (`d_tk1_idx`, constant memory), which removes key-schedule arithmetic from the encryption loop.
- **Search loop.** The key space is processed in batches (2^28 keys per kernel launch, 256 threads per block by default). A zero-copy (mapped, pinned) "found" flag lets the host stop as soon as a thread reports a matching key.
- **Register pressure.** Register usage is capped with `-maxrregcount` to avoid spilling to local memory and to keep occupancy high; Nsight Systems / Nsight Compute were used to check utilization and confirm the absence of warp divergence. `rectangle_VL_bitslice_gpu.cu` contains NVTX ranges for Nsight Systems.
- **Baselines.** The naive and table-based GPU versions exist to quantify what bitslicing gains: table lookups suffer from shared-memory bank conflicts, and the naive port from uncoalesced memory access and divergence.

## Repository layout

```
.
├── README.md
├── LICENSE
├── docs/
│   └── thesis.pdf                      # full thesis (Ukrainian)
└── src/
    ├── rectangle/
    │   ├── cpu/
    │   │   ├── rectangle_bitslice_cpu.c
    │   │   └── rectangle_AVX2_cpu.c
    │   └── gpu/
    │       ├── rectangle_naive_gpu.cu
    │       ├── rectangle_table_gpu.cu      # needs tables_RECTANGLE.inc (same folder)
    │       ├── rectangle_bitslice_gpu.cu
    │       ├── rectangle_VL_bitslice_gpu.cu
    │       └── tables_RECTANGLE.inc
    └── skinny/
        └── gpu/
            ├── skinny_table_gpu.cu         # needs tables_SKINNY.inc (same folder)
            ├── skinny_VL_bitslice_gpu.cu
            └── tables_SKINNY.inc
```

## Build

Requirements: Windows x64, Visual Studio (MSVC) build tools, CUDA Toolkit (developed with 13.1), and a CUDA-capable GPU. The programs currently use Windows-specific APIs (`windows.h`, `QueryPerformanceCounter` in the CPU versions), so they are not yet portable to Linux.

Example commands from an *x64 Native Tools Command Prompt for VS*:

```bat
:: CPU variants
cl /O2 /openmp src\rectangle\cpu\rectangle_bitslice_cpu.c
cl /O2 /openmp /arch:AVX2 src\rectangle\cpu\rectangle_AVX2_cpu.c

:: GPU variants (sm_75 = Turing, e.g. GTX 1650; adjust for your GPU)
nvcc -O3 -arch=sm_75 src\rectangle\gpu\rectangle_VL_bitslice_gpu.cu -o rectangle_vl_bitslice.exe
nvcc -O3 -arch=sm_75 src\skinny\gpu\skinny_VL_bitslice_gpu.cu -o skinny_vl_bitslice.exe
```

For the benchmark numbers above, an additional `-maxrregcount=<N>` register limit was applied to the GPU kernels. Tune `<N>` for your GPU and check the effect on occupancy.

## Usage

- `rectangle_VL_bitslice_gpu` and `skinny_VL_bitslice_gpu` ask for an operation mode:
  1. **KAT verification mode.** Encrypts an all-zero plaintext under a known key, then searches a small key range and must recover that key. The printed ciphertext can be compared with the reference vectors in Table 4.1 of the thesis.
  2. **Benchmark mode.** Scans the full benchmark range and prints the elapsed time and throughput as `2^x keys/sec`.
- The other programs run the benchmark directly.
- The size of the scanned key range is set by the `power` constant in each `main()` (range = 2^power keys). Most programs scan 2^32 keys in benchmark mode; `rectangle_bitslice_cpu.c` and `rectangle_naive_gpu.cu` are set to 2^16 keys by default.

## Estimated security margin

The key search is embarrassingly parallel and each GPU works independently from registers, so measured single-GPU throughput can be extrapolated linearly. **These figures are theoretical extrapolations, not measurements**; the H100 row additionally assumes a ≥ 32× per-GPU speedup over the laptop GPU.

Time to exhaust the *full* key space:

| Hardware (assumed) | SKINNY-64/64 (2^64 keys) | RECTANGLE-80 (2^80 keys) |
|---|---|---|
| 1 × GTX 1650 Mobile (measured rate) | ≈ 133 years | ≈ 18.6 million years |
| 10,000 × GTX 1650-class GPUs | ≈ 4.9 days | ≈ 1,858 years |
| 1,024 × NVIDIA H100 (assumed ≥ 32× per GPU) | ≈ 36 hours | ≈ 522 years |

A 64-bit key is therefore within reach of an attacker with cluster-scale GPU resources, whereas an 80-bit key retains a substantial margin today. The thesis discusses further scaling scenarios.

## Limitations

- Measurements come from one laptop GPU and one CPU; results on other hardware will differ.
- Throughput is measured on reduced key ranges, not on complete 2^64 / 2^80 searches.
- Windows-only for now (see [Build](#build)).
- Cluster and H100 figures are model-based extrapolations that assume perfect linear scaling.

## References

1. W. Zhang, Z. Bao, D. Lin, V. Rijmen, B. Yang, I. Verbauwhede. *RECTANGLE: a bit-slice lightweight block cipher suitable for multiple platforms.* Sci. China Inf. Sci. 58(12), 2015. doi:10.1007/s11432-015-5459-7
2. C. Beierle et al. *The SKINNY Family of Block Ciphers and Its Low-Latency Variant MANTIS.* CRYPTO 2016. doi:10.1007/978-3-662-53008-5_5
3. C. Tezcan, G. Leander. *GPU Assisted Brute Force Cryptanalysis of GPRS, GSM, RFID, and TETRA.* IACR ToSC 2025(1), 309–327. doi:10.46586/tosc.v2025.i1.309-327
4. C. Tezcan. *Key lengths revisited: GPU-based brute force cryptanalysis of DES, 3DES, and PRESENT.* J. Syst. Archit. 124, 2022. doi:10.1016/j.sysarc.2022.102402

## License

MIT, see [LICENSE](LICENSE).

## Author

Danylo Romenskyi, [github.com/D0wnfal1](https://github.com/D0wnfal1)
