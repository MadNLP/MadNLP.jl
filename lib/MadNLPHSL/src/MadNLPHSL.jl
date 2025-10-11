module MadNLPHSL

import MadNLP:
    MadNLP,
    @kwdef,
    MadNLPLogger,
    @debug,
    @warn,
    @error,
    AbstractOptions,
    AbstractLinearSolver,
    set_options!,
    SparseMatrixCSC,
    SubVector,
    SymbolicException,
    FactorizationException,
    SolveException,
    InertiaException,
    introduce,
    factorize!,
    solve!,
    improve!,
    is_inertia,
    inertia,
    findIJ,
    nnz,
    get_tril_to_full,
    transfer!,
    input_type,
    _madnlp_unsafe_wrap,
    is_supported,
    default_options

import HSL:
    HSL,
    Mc68Control,
    Mc68Info,
    Ma77Control,
    Ma77Info,
    Ma86Control,
    Ma86Info,
    Ma97Control,
    Ma97Info

import LinearAlgebra
import PrecompileTools: @setup_workload, @compile_workload

include("common.jl")
include("ma27.jl")
include("ma57.jl")
include("ma77.jl")
include("ma86.jl")
include("ma97.jl")

@setup_workload begin
    nlp = MadNLP.HS15Model()    
    @compile_workload begin
        for linear_solver in (Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver)
            try 
                MadNLP.madnlp(nlp; linear_solver, print_level=MadNLP.ERROR)
            catch e
                Base.@warn """
Failed to precompile MadNLPHSL. This may be caused by the absence of installed HSL_jll.jl package.
Please obtain HSL_jll.jl from https://licences.stfc.ac.uk/products/Software/HSL/libhsl, and

] dev path/to/HSL_jll.jl
"""
                break
            end
        end
    end
end

export Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver

# re-export MadNLP, including deprecated names
for name in names(MadNLP, all = true)
    if Base.isexported(MadNLP, name)
        @eval using MadNLP: $(name)
        @eval export $(name)
    end
end

end # module
