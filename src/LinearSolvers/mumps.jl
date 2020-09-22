# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module Mumps

import ..MadNLP:
    SVector, setindex, MPI,
    @with_kw, getlogger, register, setlevel!, debug, warn, error,
    SparseMatrixCSC, SubVector, StrideOneVector, libmumps, 
    SymbolicException,FactorizationException,SolveException,InertiaException,
    AbstractOptions, AbstractLinearSolver, set_options!,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, findIJ, nnz

const LOGGER=getlogger(@__MODULE__)
__init__() = register(LOGGER)
const INPUT_MATRIX_TYPE = :csc

@with_kw mutable struct Options <: AbstractOptions
    mumps_dep_tol::Float64 = 0.
    mumps_mem_percent::Int = 1000
    mumps_permuting_scaling::Int = 7
    mumps_pivot_order::Int = 7
    mumps_pivtol::Float64 = 1e-6
    mumps_pivtolmax::Float64 = .1
    mumps_scaling::Int = 77
    mumps_log_level::String = ""
end

mutable struct Struc
    sym::Cint      
    par::Cint      
    job::Cint
    
    comm_fortran::Cint
    
    icntl::SVector{40,Cint}
    cntl::SVector{15,Cdouble}
    
    n::Cint      
    nz_alloc::Cint      
    
    nz::Cint      
    irn::Ptr{Cint}
    jcn::Ptr{Cint}
    a::Ptr{Cdouble}
    
    nz_loc::Cint      
    irn_loc::Ptr{Cint}
    jcn_loc::Ptr{Cint}
    a_loc::Ptr{Cdouble}
    
    nelt::Cint      
    eltptr::Ptr{Cint}
    eltvar::Ptr{Cint}
    a_elt::Ptr{Cdouble}
    
    perm_in::Ptr{Cint}
    
    sym_perm::Ptr{Cint}
    uns_perm::Ptr{Cint}
    
    colsca::Ptr{Cdouble}
    rowsca::Ptr{Cdouble}
    
    rhs::Ptr{Cdouble}
    redrhs::Ptr{Cdouble}
    rhs_sparse::Ptr{Cdouble}
    sol_loc::Ptr{Cdouble}
    
    irhs_sparse::Ptr{Cint}
    irhs_ptr::Ptr{Cint}
    isol_loc::Ptr{Cint}
    
    nrhs::Cint
    lrhs::Cint
    lredrhs::Cint
    nz_rhs::Cint
    lsol_loc::Cint
    
    schur_mloc::Cint
    schur_nloc::Cint
    schur_lld::Cint
    
    mblock::Cint
    nblock::Cint
    nprow::Cint
    npcol::Cint

    info::SVector{40,Cint}
    infog::SVector{40,Cint}
    rinfo::SVector{40,Cdouble}
    rinfog::SVector{40,Cdouble}
    
    deficiency::Cint      
    pivnul_list::Ptr{Cint}
    mapping::Ptr{Cint}
    
    size_schur::Cint      
    listvar_schur::Ptr{Cint}
    schur::Ptr{Cdouble}
    
    instance_number::Cint      
    wk_user::Ptr{Cdouble}
    
    version_number::SVector{16,Cchar}
    
    ooc_tmpdir::SVector{256,Cchar}
    ooc_prefix::SVector{64,Cchar}
    
    write_problem::SVector{256,Cchar}
    lwk_user::Cint      
end

Struc()=Struc(0,0,0,0,SVector{40}(zeros(Int32,40)),SVector{15}(zeros(15)),0,0,0,C_NULL,C_NULL,C_NULL,0,C_NULL,C_NULL,C_NULL,0,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,0,0,0,0,0,0,0,0,0,0,0,0,SVector{40}(zeros(Int32,40)),SVector{40}(zeros(Int32,40)),SVector{40}(zeros(40)),SVector{40}(zeros(40)),0,C_NULL,C_NULL,0,C_NULL,C_NULL,0,C_NULL,SVector{16}(zeros(Int8,16)),SVector{256}(zeros(Int8,256)),SVector{64}(zeros(Int8,64)),SVector{256}(zeros(Int8,256)),0)

mutable struct Solver <: AbstractLinearSolver
    csc::SparseMatrixCSC{Float64,Int32}
    I::Vector{Int32}
    J::Vector{Int32}
    sym_perm::Vector{Int32}
    pivnul_list::Vector{Int32}
    mumps_struc::Struc
    is_singular::Bool
    opt::Options
end

dmumps_c(mumps_struc::Struc)=ccall(
    (:dmumps_c, libmumps),
    Cvoid,
    (Ref{Struc},),
    mumps_struc)

function Solver(csc::SparseMatrixCSC{Float64,Int32};
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Options(),
                kwargs...)
    
    set_options!(opt,option_dict,kwargs)
    opt.mumps_log_level=="" || setlevel!(LOGGER,opt.mumps_log_level)
        
    I,J = findIJ(csc)
    sym_perm = zeros(Int32,csc.n)
    pivnul_list = zeros(Int32,csc.n)
    MPI.Initialized() || MPI.Init()
    
    mumps_struc = Struc()
    
    mumps_struc.sym=2
    mumps_struc.par=1
    mumps_struc.job=-1
    mumps_struc.comm_fortran = -987654 # MPI.COMM_WORLD.val

    dmumps_c(mumps_struc)
    
    mumps_struc.n = csc.n;
    mumps_struc.nz= nnz(csc);
    mumps_struc.a = pointer(csc.nzval)
    mumps_struc.irn = pointer(I)
    mumps_struc.jcn = pointer(J)
    mumps_struc.sym_perm = pointer(sym_perm)
    mumps_struc.pivnul_list = pointer(pivnul_list)
    
    # symbolic factorization
    mumps_struc.job = 1;
    
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

    a = copy(csc.nzval) # would there be a better way?
    csc.nzval.=1
    
    dmumps_c(mumps_struc);    
    mumps_struc.info[1] < 0 && throw(SymbolicException())

    csc.nzval.=a
    
    return Solver(csc,I,J,sym_perm,pivnul_list,mumps_struc,false,opt)
end

function factorize!(M::Solver)
    M.is_singular = false
    M.mumps_struc.job = 2;
    cnt = 0
    while true
        dmumps_c(M.mumps_struc)
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

function solve!(M::Solver,rhs::StrideOneVector{Float64})
    M.is_singular && return rhs
    M.mumps_struc.rhs = pointer(rhs)
    M.mumps_struc.job = 3
    dmumps_c(M.mumps_struc)
    M.mumps_struc.info[1] < 0 && throw(SolveException())
    return rhs
end

is_inertia(::Solver) = true
function inertia(M::Solver)
    return (M.csc.n-M.is_singular-M.mumps_struc.infog[12],
            M.is_singular,
            M.mumps_struc.infog[12])
end


function improve!(M::Solver)
    if M.mumps_struc.cntl[1] == M.opt.mumps_pivtolmax
        debug(LOGGER,"improve quality failed.")
        return false
    end
    M.mumps_struc.cntl = setindex(M.mumps_struc.cntl,min(M.opt.mumps_pivtolmax,M.mumps_struc.cntl[1]^.5),1)
    debug(LOGGER,"improved quality: pivtol = $(M.mumps_struc.cntl[1])")
    return true
end

introduce(::Solver)="mumps"

end # module

# forgiving names
const mumps=Mumps;
const MUMPS=Mumps;

