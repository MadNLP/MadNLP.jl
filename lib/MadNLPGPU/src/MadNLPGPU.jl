module MadNLPGPU

# CUDA
import CUDA: CUBLAS, CUSOLVER, CuVector, CuMatrix, CuArray, toolkit_version, R_64F, has_cuda
# Kernels
using KernelAbstractions
using CUDAKernels
using LinearAlgebra
using MadNLP

import MadNLP:
    @kwdef, Logger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, set_options!, MadNLPLapackCPU,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!


include("kernels.jl")

if has_cuda()
    include("lapackgpu.jl")
    export MadNLPLapackGPU
end

end # module
