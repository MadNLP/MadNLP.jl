module MadNLPMumps

import MUMPS

import MadNLP:
    MadNLP, parsefile,
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    SparseMatrixCSC, SubVector,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    AbstractOptions, AbstractLinearSolver, set_options!, input_type, default_options,
    introduce, factorize!, solve!, improve!, is_inertia, is_supported, inertia, findIJ, nnz
import LinearAlgebra, OpenBLAS32_jll

function __init__()
    if VERSION ≥ v"1.9"
        config = LinearAlgebra.BLAS.lbt_get_config()
        if !any(lib -> lib.interface == :lp64, config.loaded_libs)
            LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
        end
    end
end

version = string(pkgversion(@__MODULE__))

setindex(tup,a,n) = (tup[1:n-1]...,a,tup[n+1:end]...)
tzeros(n) = tuple((0 for i=1:n)...)

@kwdef mutable struct MumpsOptions <: AbstractOptions
    mumps_dep_tol::Float64 = 0.
    mumps_mem_percent::Int = 1000
    mumps_permuting_scaling::Int = 7
    mumps_pivot_order::Int = 7
    mumps_pivtol::Float64 = 1e-6
    mumps_pivtolmax::Float64 = .1
    mumps_scaling::Int = 77
end

mutable struct MumpsSolver{T} <: AbstractLinearSolver{T}
    csc::SparseMatrixCSC{T,Int32}
    I::Vector{Int32}
    J::Vector{Int32}
    sym_perm::Vector{Int32}
    pivnul_list::Vector{Int32}
    mumps_struc::MUMPS.Mumps
    is_singular::Bool
    opt::MumpsOptions
    logger::MadNLPLogger
end

function MumpsSolver(csc::SparseMatrixCSC{T,Int32};
    opt=MumpsOptions(), logger=MadNLPLogger(),
) where T

    I,J = findIJ(csc)
    sym_perm = zeros(Int32,csc.n)
    pivnul_list = zeros(Int32,csc.n)

    icntl = MUMPS.default_icntl[:]
    mumps_struc = MUMPS.Mumps{T}(MUMPS.mumps_symmetric, icntl, MUMPS.default_cntl64)

    mumps_struc.n = csc.n;
    mumps_struc.nnz= nnz(csc);
    mumps_struc.a = pointer(csc.nzval)
    mumps_struc.irn = pointer(I)
    mumps_struc.jcn = pointer(J)
    mumps_struc.sym_perm = pointer(sym_perm)
    mumps_struc.pivnul_list = pointer(pivnul_list)

    mumps_struc.icntl = setindex(mumps_struc.icntl,0,2)
    mumps_struc.icntl = setindex(mumps_struc.icntl,0,3)
    mumps_struc.icntl = setindex(mumps_struc.icntl,0,4)
    mumps_struc.icntl = setindex(mumps_struc.icntl,opt.mumps_permuting_scaling,6)
    mumps_struc.icntl = setindex(mumps_struc.icntl,opt.mumps_pivot_order,7)
    mumps_struc.icntl = setindex(mumps_struc.icntl,opt.mumps_scaling,8)
    mumps_struc.icntl = setindex(mumps_struc.icntl,0,10)
    mumps_struc.icntl = setindex(mumps_struc.icntl,1,13)
    mumps_struc.icntl = setindex(mumps_struc.icntl,opt.mumps_mem_percent,14)

    mumps_struc.cntl = setindex(mumps_struc.cntl,opt.mumps_pivtol,1)


    mumps_struc.job = 1
    a = copy(csc.nzval) # would there be a better way?
    csc.nzval.=1
    MUMPS.invoke_mumps!(mumps_struc)
    mumps_struc.info[1] < 0 && throw(SymbolicException())

    csc.nzval.=a

    M = MumpsSolver{T}(csc, I, J, sym_perm, pivnul_list, mumps_struc, false, opt, logger)
    finalizer(finalize,M)

    return M
end

function factorize!(M::MumpsSolver)
    M.is_singular = false
    M.mumps_struc.job = 2
    cnt = 0
    while true
        MUMPS.invoke_mumps!(M.mumps_struc)
        if M.mumps_struc.info[1] in [-8,-9]
            cnt >= 10 && throw(FactorizationException())
            M.mumps_struc.icntl = setindex(M.mumps_struc.icntl,M.mumps_struc.icntl[14]*2.,14)
            cnt += 1
        elseif M.mumps_struc.info[1] == -10
            M.is_singular = true
            break
        elseif M.mumps_struc.info[1] < 0
            throw(FactorizationException())
        else
            break
        end
    end
    return M
end

function solve!(M::MumpsSolver{T},rhs::Vector{T}) where T
    M.is_singular && return rhs
    M.mumps_struc.rhs = pointer(rhs)
    M.mumps_struc.lrhs = length(rhs)
    M.mumps_struc.nrhs = 1
    M.mumps_struc.job = 3
    MUMPS.invoke_mumps!(M.mumps_struc)
    M.mumps_struc.info[1] < 0 && throw(SolveException())
    return rhs
end

is_inertia(::MumpsSolver) = true
function inertia(M::MumpsSolver)
    return (M.csc.n-M.is_singular-M.mumps_struc.infog[12],
            M.is_singular,
            M.mumps_struc.infog[12])
end


function improve!(M::MumpsSolver)
    if M.mumps_struc.cntl[1] == M.opt.mumps_pivtolmax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.mumps_struc.cntl = setindex(M.mumps_struc.cntl,min(M.opt.mumps_pivtolmax,M.mumps_struc.cntl[1]^.5),1)
    @debug(M.logger,"improved quality: pivtol = $(M.mumps_struc.cntl[1])")
    return true
end

function finalize(M::MumpsSolver)
    MUMPS.finalize(M.mumps_struc)
end

introduce(::MumpsSolver) = "mumps"
input_type(::Type{MumpsSolver}) = :csc
default_options(::Type{MumpsSolver}) = MumpsOptions()
is_supported(::Type{MumpsSolver},::Type{Float32}) = true
is_supported(::Type{MumpsSolver},::Type{Float64}) = true

export MumpsSolver

# re-export MadNLP, including deprecated names
for name in names(MadNLP, all=true)
    if Base.isexported(MadNLP, name)
        @eval using MadNLP: $(name)
        @eval export $(name)
    end
end

end # module
