# CuMadNLP.jl

IPM-specific GPU support for MadNLP on CUDA: GPU methods of the interior-point
error/restoration kernels, the GPU bound counter and scaling getters, and the
GPU-default `MadNLPOptions` constructor. Built on `MadCoreCUDA`. Opt-in package
(`using CuMadNLP` to solve models with CUDA arrays).
