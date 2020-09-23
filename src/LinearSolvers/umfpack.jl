# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module Umfpack

import ..MadNLP:
    @with_kw, Logger, @debug, @warn, @error,
    SubVector, StrideOneVector, SparseMatrixCSC, get_tril_to_full,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    AbstractOptions, AbstractLinearSolver, set_options!, UMFPACK,
    introduce, factorize!, solve!, mul!, improve!, is_inertia, inertia

const INPUT_MATRIX_TYPE = :csc

const umfpack_default_ctrl = copy(UMFPACK.umf_ctrl)
const umfpack_default_info = copy(UMFPACK.umf_info)

@with_kw mutable struct Options <: AbstractOptions
    umfpack_pivtol::Float64 = 1e-4
    umfpack_pivtolmax::Float64 = 1e-1
    umfpack_sym_pivtol::Float64 = 1e-3
    umfpack_block_size::Float64 = 16
    umfpack_strategy::Float64 = 2.
end

mutable struct Solver <: AbstractLinearSolver
    inner::UMFPACK.UmfpackLU
    tril::SparseMatrixCSC{Float64}
    full::SparseMatrixCSC{Float64}
    tril_to_full_view::SubVector{Float64}
    
    p::Vector{Float64}

    ctrl::Vector{Float64}
    info::Vector{Float64}

    opt::Options
    logger::Logger
end

umfpack_di_numeric(colptr::StrideOneVector{Int32},rowval::StrideOneVector{Int32},
                   nzval::StrideOneVector{Float64},symbolic::Ptr{Nothing},
                   tmp::Vector{Ptr{Nothing}},ctrl::Vector{Float64},
                   info::Vector{Float64}) = ccall(
                       (:umfpack_di_numeric,:libumfpack),
                       Int32,
                       (Ptr{Int32},Ptr{Int32},Ptr{Float64},Ptr{Cvoid},Ptr{Cvoid},
                        Ptr{Float64},Ptr{Float64}),
                       colptr,rowval,nzval,symbolic,tmp,ctrl,info)

function Solver(csc::SparseMatrixCSC;
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Options(),logger=Logger(),
                kwargs...)
    
    set_options!(opt,option_dict,kwargs)
    
    p = Vector{Float64}(undef,csc.n)
    full,tril_to_full_view = get_tril_to_full(csc)
    
    full.colptr.-=1; full.rowval.-=1
    
    inner = UMFPACK.UmfpackLU(C_NULL,C_NULL,full.n,full.n,full.colptr,full.rowval,full.nzval,0)
    UMFPACK.finalizer(UMFPACK.umfpack_free_symbolic,inner)
    UMFPACK.umfpack_symbolic!(inner)
    ctrl = copy(umfpack_default_ctrl)
    info = copy(umfpack_default_info)
    ctrl[4]=opt.umfpack_pivtol
    ctrl[12]=opt.umfpack_sym_pivtol
    ctrl[5]=opt.umfpack_block_size
    ctrl[6]=opt.umfpack_strategy
    return Solver(inner,csc,full,tril_to_full_view,p,ctrl,info,opt,logger)
end
function factorize!(M::Solver)
    UMFPACK.umfpack_free_numeric(M.inner)
    M.full.nzval.=M.tril_to_full_view
    tmp = Vector{Ptr{Cvoid}}(undef, 1)
    status=umfpack_di_numeric(M.inner.colptr,M.inner.rowval,M.inner.nzval,
                              M.inner.symbolic,tmp,M.ctrl,M.info)
    M.inner.numeric = tmp[1]
    M.inner.status = status
    return M
end
function solve!(M::Solver,rhs::StrideOneVector{Float64})
    rhs.=UMFPACK.solve!(M.p,M.inner,rhs,1)
    return rhs
end
is_inertia(::Solver) = false
inertia(M::Solver) = throw(InertiaException())

function improve!(M::Solver)
    if M.ctrl[4] == M.opt.umfpack_pivtolmax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.ctrl[4] = min(M.opt.umfpack_pivtolmax,M.ctrl[4]^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.ctrl[4])")
    return true

    return false
end
introduce(::Solver)="umfpack"

end # module
    
# forgiving names
const umfpack=Umfpack;
