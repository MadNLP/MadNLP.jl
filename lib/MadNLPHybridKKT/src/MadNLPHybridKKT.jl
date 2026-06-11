module MadNLPHybridKKT

# Hybrid (HyKKT / condensed-space) KKT system for MadNLP, incorporated from
# MadNLP/HybridKKT.jl. Built on the lightweight MadCore (KKT infra) + MadNLPCore
# (IPM) instead of the full MadNLP, and kept CUDA-free: GPU code dispatches on
# AbstractGPUVector and runs on whatever KernelAbstractions backend the arrays
# live on (data-transfer kernels from MadCoreGPUArrays, transfer/diag kernels from
# MadCoreKernelAbstractions). MadNLP imports this package.

using LinearAlgebra
using SparseArrays
using Printf

import Krylov
import NLPModels

# Lightweight core (KKT infrastructure) reexported by MadNLPCore, plus the IPM.
using MadNLPCore
import MadCore: SparseMatrixCOO, full,
    create_kkt_system, build_kkt!, solve_kkt!, compress_jacobian!, compress_hessian!,
    get_jacobian, jtprod!, is_inertia_correct, initialize!
import MadNLPCore: solve_refine_wrapper!

# Backend-agnostic GPU pieces (no hard CUDA dependency).
import GPUArraysCore: AbstractGPUVector
import KernelAbstractions
import KernelAbstractions: get_backend
import MadCoreKernelAbstractions
import MadCoreGPUArrays: index_copy!, fixed!, transfer_coef!

include("utils.jl")
include("kkt.jl")

export HybridCondensedKKTSystem

end # module MadNLPHybridKKT
