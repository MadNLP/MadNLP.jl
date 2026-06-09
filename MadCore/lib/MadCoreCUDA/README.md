# MadCoreCUDA.jl

CUDA backend for MadCore: `LapackCUDASolver` (cuSOLVER), `CUDSSSolver` (cuDSS),
and the GPU Schur-complement solver. Builds on MadCore + MadCoreKernelAbstractions.
IPM-specific GPU integration lives in `MadNLP/lib/CuMadNLP`.
