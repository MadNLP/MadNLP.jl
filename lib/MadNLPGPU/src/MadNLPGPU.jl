module MadNLPGPU

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric, mul!, norm, dot

# GPU abstractions
import GPUArraysCore: AbstractGPUVector, AbstractGPUMatrix, AbstractGPUArray, @allowscalar
# Kernels
import KernelAbstractions: @kernel, @index, synchronize, @Const, get_backend

import MadNLP: NLPModels
import MadNLP
import MadNLP:
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type, is_supported, default_options,
    setup_cholesky!, setup_lu!, setup_qr!, setup_evd!, setup_bunchkaufman!,
    factorize_cholesky!, factorize_lu!, factorize_qr!, factorize_evd!, factorize_bunchkaufman!,
    solve_cholesky!, solve_lu!, solve_qr!, solve_evd!, solve_bunchkaufman!

# AMD and Metis
import AMD, Metis

include("KKT/kernels_dense.jl")
include("KKT/kernels_sparse.jl")
include("KKT/kernels_qn.jl")
include("utils.jl")
include("KKT/gpu_dense.jl")
include("KKT/gpu_sparse.jl")
include("KKT/gpu_qn.jl")

global LapackCUDASolver
global CUDSSSolver
global LapackROCmSolver
export LapackCUDASolver, CUDSSSolver, LapackROCmSolver

# re-export MadNLP, including deprecated names
for name in names(MadNLP, all=true)
    if Base.isexported(MadNLP, name)
        @eval using MadNLP: $(name)
        @eval export $(name)
    end
end

end # module
