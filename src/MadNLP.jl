module MadNLP

import Pkg.TOML: parsefile
import Printf: @sprintf
import LinearAlgebra: BLAS, Adjoint, Symmetric, mul!, ldiv!, norm, dot, diagind, normInf, transpose!, issuccess
import LinearAlgebra: cholesky, qr, lu, cholesky!, axpy!
import LinearAlgebra.BLAS: symv!, ger!, libblastrampoline, BlasInt, @blasfunc
import SparseArrays: SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, sparse, getcolptr, rowvals, nnz, nonzeros
import Base: Base, string, show, print, size, getindex, copyto!, @kwdef
import SuiteSparse: UMFPACK, CHOLMOD
import NLPModels: NLPModels, AbstractNLPModel, AbstractNLPModel, AbstractNLPModelMeta,
    finalize, AbstractNLPModel, obj, grad!, cons!, jac_coord!, hess_coord!, hess_structure!, jac_structure!, jtprod!
    get_nvar, get_ncon, get_minimize, get_x0, get_y0, get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon
import SolverCore: solve!, getStatus, AbstractOptimizationSolver, AbstractExecutionStats
import LDLFactorizations
import MUMPS_seq_jll, OpenBLAS32_jll
import PrecompileTools: @setup_workload, @compile_workload   

export MadNLPSolver, MadNLPOptions, UmfpackSolver, LDLSolver, CHOLMODSolver, LapackCPUSolver, MumpsSolver, MadNLPExecutionStats, madnlp, solve!

function __init__()
    config = BLAS.lbt_get_config()
    if !any(lib -> lib.interface == :lp64, config.loaded_libs)
        BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end

    # Misteriously not compiled functions
    precompile(Tuple{typeof(Base.iterate), Base.Dict{Symbol, Any}})
    precompile(Tuple{typeof(Base.fill!), Array{Float64, 1}, Float64})
    precompile(Tuple{typeof(Base.view), Base.BitArray{1}, Array{Int64, 1}})
    precompile(Tuple{typeof(Base.view), Array{Int64, 1}, Base.UnitRange{Int64}})
    precompile(Tuple{typeof(Base.view), Array{Float64, 1}, Base.UnitRange{Int64}})
    precompile(Tuple{typeof(Base.:(+)), Vararg{Int64, 5}})        
    precompile(Tuple{Type{Array{Float64, 1}}, UndefInitializer, Int64})                
    precompile(Tuple{typeof(Base.Broadcast.broadcasted), typeof(Base.:(+)), Array{Int32, 1}, Int64})        
    precompile(Tuple{typeof(Base.:(+)), Vararg{Int64, 4}})        
    precompile(Tuple{typeof(Base.getindex), Array{Float64, 1}, Base.UnitRange{Int64}}) 
    precompile(Tuple{typeof(Base.getindex), Base.RefValue{Float64}})                   
    precompile(Tuple{typeof(Base.copy), Array{Float64, 1}})       
    precompile(Tuple{typeof(Base.sum), Base.BitArray{1}})         
    precompile(Tuple{typeof(Base.length), Array{Float64, 1}})     
    precompile(Tuple{typeof(Base.:(>)), Float64, Float64})        
    precompile(Tuple{typeof(Base.:(-)), Int64, Float64})          
    precompile(Tuple{typeof(Base.:(>=)), Float64, Float64})       
    precompile(Tuple{typeof(Base.:(var"==")), Bool, Int64})
end


# Version info
version() = string(pkgversion(@__MODULE__))
introduce() = "\033[34mMad\033[31mN\033[32mL\033[35mP\033[0m version v$(version())"

include("enums.jl")
include("utils.jl")
include("matrixtools.jl")
include("nlpmodels.jl")
include("quasi_newton.jl")
include(joinpath("KKT", "KKTsystem.jl"))
include(joinpath("LinearSolvers","linearsolvers.jl"))
include("options.jl")
include(joinpath("IPM", "IPM.jl"))
include("extension_templates.jl")
include("precompile.jl")

end # end module
