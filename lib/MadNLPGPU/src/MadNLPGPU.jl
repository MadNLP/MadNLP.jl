module MadNLPGPU

import LinearAlgebra
# CUDA
import CUDA: CUBLAS, CUSOLVER, CuVector, CuMatrix, CuArray, R_64F, has_cuda, @allowscalar, runtime_version
# Kernels
import KernelAbstractions: @kernel, @index, wait, Event
import CUDAKernels: CUDADevice

import MadNLP

import MadNLP:
    @kwdef, Logger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type



include("kernels.jl")

if has_cuda()
    include("lapackgpu.jl")
    export LapackGPUSolver
end
include("interface.jl")

end # module
