module MadNLP

using Reexport
# The interior-point solver itself now lives in MadNLPCore (built on the
# lightweight MadCore). MadNLP is the batteries-included meta-package: it bundles
# the default linear-solver backends and specializes the bare-solver defaults.
@reexport using MadNLPCore
@reexport using MadCoreLDLFactorizations
@reexport using MadCoreMUMPS
@reexport using MadCoreSuiteSparse
@reexport using MadCoreHSL
@reexport using MadCorePardiso

import MadNLPCore: madnlp, MadNLPOptions, is_dense_callback, get_tolerance
import NLPModels
import NLPModels: AbstractNLPModel

using PrecompileTools: @setup_workload, @compile_workload

export madsuite, default_sparse_solver

madsuite(::Val{:madnlp}, args...; kwargs...) = madnlp(args...; kwargs...)

# MadNLPCore defaults the sparse linear solver to the no-op DummyLinearSolver.
# MadNLP bundles MadCoreMUMPS, so it provides `default_sparse_solver` and a
# Vector-specialized MadNLPOptions constructor (a *more specific* method than
# MadNLPCore's generic one — a method addition, not a precompile-breaking
# overwrite) that uses it. The GPU backends specialize on CuVector/ROCVector.
default_sparse_solver(nlp) = MumpsSolver

function MadNLPOptions{T}(
    nlp::AbstractNLPModel{T, VT};
    dense_callback = is_dense_callback(nlp),
    callback = dense_callback ? DenseCallback : SparseCallback,
    kkt_system = dense_callback ? DenseCondensedKKTSystem : SparseKKTSystem,
    linear_solver = dense_callback ? LapackCPUSolver : default_sparse_solver(nlp),
    tol = get_tolerance(T, kkt_system),
) where {T, VT <: Vector{T}}
    return MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
    )
end

include("precompile.jl")

global Optimizer

end # end module
