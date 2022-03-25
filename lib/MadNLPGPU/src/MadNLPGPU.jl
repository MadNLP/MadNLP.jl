module MadNLPGPU

import LinearAlgebra
# CUDA
import CUDA
import CUDA: CUBLAS, CUSOLVER, CuVector, CuMatrix, CuArray, R_64F, has_cuda
# Kernels
import KernelAbstractions: @kernel, @index, wait
import CUDAKernels: CUDADevice

import MadNLP

import MadNLP:
    @kwdef, Logger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!, MadNLPLapackCPU,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!


include("kernels.jl")

if has_cuda()
    include("lapackgpu.jl")
    export MadNLPLapackGPU
end

end # module
