module MadNLPMumps

import StaticArrays: SVector, setindex
import MUMPS_seq_jll
import MadNLP:
    parsefile, dlopen,
    @kwdef, Logger, @debug, @warn, @error,
    SparseMatrixCSC, SubVector, StrideOneVector, 
    SymbolicException,FactorizationException,SolveException,InertiaException,
    AbstractOptions, AbstractLinearSolver, set_options!,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, findIJ, nnz

const INPUT_MATRIX_TYPE = :csc
const version = parsefile(joinpath(dirname(pathof(MUMPS_seq_jll)),"..","Project.toml"))["version"]

@kwdef mutable struct Options <: AbstractOptions
    mumps_dep_tol::Float64 = 0.
    mumps_mem_percent::Int = 1000
    mumps_permuting_scaling::Int = 7
    mumps_pivot_order::Int = 7
    mumps_pivtol::Float64 = 1e-6
    mumps_pivtolmax::Float64 = .1
    mumps_scaling::Int = 77
end

if version == "5.3.5+0"
    @kwdef mutable struct Struc
        sym::Cint = 0
        par::Cint = 0
        job::Cint = 0

        comm_fortran::Cint = 0

        icntl::SVector{60,Cint} = zeros(60)
        keep::SVector{500,Cint} = zeros(500)
        cntl::SVector{15,Cdouble} = zeros(15)
        dkeep::SVector{230,Cdouble} = zeros(230)
        keep8::SVector{150,Int64} = zeros(150)
        n::Cint = 0
        nblk::Cint = 0
        
        nz_alloc::Cint = 0

        nz::Cint = 0
        nnz::Int64 = 0
        irn::Ptr{Cint} = C_NULL
        jcn::Ptr{Cint} = C_NULL
        a::Ptr{Cdouble} = C_NULL

        nz_loc::Cint = 0
        nnz_loc::Int64 = 0
        irn_loc::Ptr{Cint} = C_NULL
        jcn_loc::Ptr{Cint} = C_NULL
        a_loc::Ptr{Cdouble} = C_NULL ###

        nelt::Cint = 0
        eltptr::Ptr{Cint} = C_NULL
        eltvar::Ptr{Cint} = C_NULL
        a_elt::Ptr{Cdouble} = C_NULL

        blkptr::Ptr{Cint} = C_NULL
        blkvar::Ptr{Cint} = C_NULL

        perm_in::Ptr{Cint} = C_NULL

        sym_perm::Ptr{Cint} = C_NULL
        uns_perm::Ptr{Cint} = C_NULL

        colsca::Ptr{Cdouble} = C_NULL
        rowsca::Ptr{Cdouble} = C_NULL
        colsca_from_mumps::Cint = 0
        rowsca_from_mumps::Cint = 0

        rhs::Ptr{Cdouble} = C_NULL
        redrhs::Ptr{Cdouble} = C_NULL
        rhs_sparse::Ptr{Cdouble} = C_NULL
        sol_loc::Ptr{Cdouble} = C_NULL
        rhs_loc::Ptr{Cdouble} = C_NULL

        irhs_sparse::Ptr{Cint} = C_NULL
        irhs_ptr::Ptr{Cint} = C_NULL
        isol_loc::Ptr{Cint} = C_NULL
        irhs_loc::Ptr{Cint} = C_NULL
        
        nrhs::Cint = 0
        lrhs::Cint = 0
        lredrhs::Cint = 0
        nz_rhs::Cint = 0
        lsol_loc::Cint = 0
        nloc_rhs::Cint = 0
        lrhs_loc::Cint = 0

        schur_mloc::Cint = 0
        schur_nloc::Cint = 0
        schur_lld::Cint = 0

        mblock::Cint = 0
        nblock::Cint = 0
        nprow::Cint = 0
        npcol::Cint = 0

        info::SVector{80,Cint} = zeros(80)
        infog::SVector{80,Cint} = zeros(80)
        rinfo::SVector{40,Cdouble} = zeros(40)
        rinfog::SVector{40,Cdouble} = zeros(40)

        deficiency::Cint = 0
        pivnul_list::Ptr{Cint} = C_NULL
        mapping::Ptr{Cint} = C_NULL

        size_schur::Cint = 0
        listvar_schur::Ptr{Cint} = C_NULL
        schur::Ptr{Cdouble} = C_NULL ##

        instance_number::Cint = 0
        wk_user::Ptr{Cdouble} = C_NULL 

        version_number::SVector{32,Cchar} = zeros(32)

        ooc_tmpdir::SVector{256,Cchar} = zeros(256)
        ooc_prefix::SVector{64,Cchar} = zeros(64)

        write_problem::SVector{256,Cchar} = zeros(256)
        lwk_user::Cint = 0

        save_dir::SVector{256,Cchar} = zeros(256)
        save_prefix::SVector{256,Cchar} = zeros(256)

        metis_options::SVector{40,Cint} = zeros(40)
    end
elseif version == "5.2.1+4"
    @kwdef mutable struct Struc
        sym::Cint = 0
        par::Cint = 0
        job::Cint = 0

        comm_fortran::Cint = 0

        icntl::SVector{60,Cint} = zeros(60)
        keep::SVector{500,Cint} = zeros(500)
        cntl::SVector{15,Cdouble} = zeros(15)
        dkeep::SVector{230,Cdouble} = zeros(230)
        keep8::SVector{150,Int64} = zeros(150)
        n::Cint = 0
        
        nz_alloc::Cint = 0

        nz::Cint = 0
        nnz::Int64 = 0
        irn::Ptr{Cint} = C_NULL
        jcn::Ptr{Cint} = C_NULL
        a::Ptr{Cdouble} = C_NULL

        nz_loc::Cint = 0
        nnz_loc::Int64 = 0
        irn_loc::Ptr{Cint} = C_NULL
        jcn_loc::Ptr{Cint} = C_NULL
        a_loc::Ptr{Cdouble} = C_NULL ###

        nelt::Cint = 0
        eltptr::Ptr{Cint} = C_NULL
        eltvar::Ptr{Cint} = C_NULL
        a_elt::Ptr{Cdouble} = C_NULL

        perm_in::Ptr{Cint} = C_NULL

        sym_perm::Ptr{Cint} = C_NULL
        uns_perm::Ptr{Cint} = C_NULL

        colsca::Ptr{Cdouble} = C_NULL
        rowsca::Ptr{Cdouble} = C_NULL
        colsca_from_mumps::Cint = 0
        rowsca_from_mumps::Cint = 0

        rhs::Ptr{Cdouble} = C_NULL
        redrhs::Ptr{Cdouble} = C_NULL
        rhs_sparse::Ptr{Cdouble} = C_NULL
        sol_loc::Ptr{Cdouble} = C_NULL
        rhs_loc::Ptr{Cdouble} = C_NULL

        irhs_sparse::Ptr{Cint} = C_NULL
        irhs_ptr::Ptr{Cint} = C_NULL
        isol_loc::Ptr{Cint} = C_NULL
        irhs_loc::Ptr{Cint} = C_NULL
        
        nrhs::Cint = 0
        lrhs::Cint = 0
        lredrhs::Cint = 0
        nz_rhs::Cint = 0
        lsol_loc::Cint = 0
        nloc_rhs::Cint = 0
        lrhs_loc::Cint = 0

        schur_mloc::Cint = 0
        schur_nloc::Cint = 0
        schur_lld::Cint = 0

        mblock::Cint = 0
        nblock::Cint = 0
        nprow::Cint = 0
        npcol::Cint = 0

        info::SVector{80,Cint} = zeros(80)
        infog::SVector{80,Cint} = zeros(80)
        rinfo::SVector{40,Cdouble} = zeros(40)
        rinfog::SVector{40,Cdouble} = zeros(40)

        deficiency::Cint = 0
        pivnul_list::Ptr{Cint} = C_NULL
        mapping::Ptr{Cint} = C_NULL

        size_schur::Cint = 0
        listvar_schur::Ptr{Cint} = C_NULL
        schur::Ptr{Cdouble} = C_NULL ##

        instance_number::Cint = 0
        wk_user::Ptr{Cdouble} = C_NULL 

        version_number::SVector{32,Cchar} = zeros(32)

        ooc_tmpdir::SVector{256,Cchar} = zeros(256)
        ooc_prefix::SVector{64,Cchar} = zeros(64)

        write_problem::SVector{256,Cchar} = zeros(256)
        lwk_user::Cint = 0

        save_dir::SVector{256,Cchar} = zeros(256)
        save_prefix::SVector{256,Cchar} = zeros(256)

        metis_options::SVector{40,Cint} = zeros(40)
    end
# elseif version == "4.10.0+0"
#     @kwdef mutable struct Struc
#         sym::Cint = 0
#         par::Cint = 0
#         job::Cint = 0

#         comm_fortran::Cint = 0

#         icntl::SVector{40,Cint} = zeros(40)
#         cntl::SVector{15,Cdouble} = zeros(15)

#         n::Cint = 0
#         nz_alloc::Cint = 0

#         nz::Cint = 0
#         irn::Ptr{Cint} = C_NULL
#         jcn::Ptr{Cint} = C_NULL
#         a::Ptr{Cdouble} = C_NULL

#         nz_loc::Cint = 0
#         irn_loc::Ptr{Cint} = C_NULL
#         jcn_loc::Ptr{Cint} = C_NULL
#         a_loc::Ptr{Cdouble} = C_NULL

#         nelt::Cint = 0
#         eltptr::Ptr{Cint} = C_NULL
#         eltvar::Ptr{Cint} = C_NULL
#         a_elt::Ptr{Cdouble} = C_NULL

#         perm_in::Ptr{Cint} = C_NULL

#         sym_perm::Ptr{Cint} = C_NULL
#         uns_perm::Ptr{Cint} = C_NULL

#         colsca::Ptr{Cdouble} = C_NULL
#         rowsca::Ptr{Cdouble} = C_NULL

#         rhs::Ptr{Cdouble} = C_NULL
#         redrhs::Ptr{Cdouble} = C_NULL
#         rhs_sparse::Ptr{Cdouble} = C_NULL
#         sol_loc::Ptr{Cdouble} = C_NULL

#         irhs_sparse::Ptr{Cint} = C_NULL
#         irhs_ptr::Ptr{Cint} = C_NULL
#         isol_loc::Ptr{Cint} = C_NULL

#         nrhs::Cint = 0
#         lrhs::Cint = 0
#         lredrhs::Cint = 0
#         nz_rhs::Cint = 0
#         lsol_loc::Cint = 0

#         schur_mloc::Cint = 0
#         schur_nloc::Cint = 0
#         schur_lld::Cint = 0

#         mblock::Cint = 0
#         nblock::Cint = 0
#         nprow::Cint = 0
#         npcol::Cint = 0

#         info::SVector{40,Cint} = zeros(40)
#         infog::SVector{40,Cint} = zeros(40)
#         rinfo::SVector{40,Cdouble} = zeros(40)
#         rinfog::SVector{40,Cdouble} = zeros(40)

#         deficiency::Cint = 0
#         pivnul_list::Ptr{Cint} = C_NULL
#         mapping::Ptr{Cint} = C_NULL

#         size_schur::Cint = 0
#         listvar_schur::Ptr{Cint} = C_NULL
#         schur::Ptr{Cdouble} = C_NULL

#         instance_number::Cint = 0
#         wk_user::Ptr{Cdouble} = C_NULL

#         version_number::SVector{16,Cchar} = zeros(16)

#         ooc_tmpdir::SVector{256,Cchar} = zeros(256)
#         ooc_prefix::SVector{64,Cchar} = zeros(64)

#         write_problem::SVector{256,Cchar} = zeros(256)
#         lwk_user::Cint = 0
#     end
else
    error("MUMPS_seq_jll version not supported")
end

mutable struct Solver <: AbstractLinearSolver
    csc::SparseMatrixCSC{Float64,Int32}
    I::Vector{Int32}
    J::Vector{Int32}
    sym_perm::Vector{Int32}
    pivnul_list::Vector{Int32}
    mumps_struc::Struc
    is_singular::Bool
    opt::Options
    logger::Logger
end

dmumps_c(mumps_struc::Struc)=ccall(
    (:dmumps_c,MUMPS_seq_jll.libdmumps),
    Cvoid,
    (Ref{Struc},),
    mumps_struc)

# this is necessary, when multi-threaded calls are made with Mumps, not to clash with MPI
mumps_lock = Threads.SpinLock()
function locked_dmumps_c(mumps_struc::Struc)
    lock(mumps_lock)
    try
        dmumps_c(mumps_struc)
    finally
        unlock(mumps_lock)
    end
end
# ---------------------------------------------------------------------------------------

function Solver(csc::SparseMatrixCSC{Float64,Int32};
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Options(),logger=Logger(),
                kwargs...)

    set_options!(opt,option_dict,kwargs)

    I,J = findIJ(csc)
    sym_perm = zeros(Int32,csc.n)
    pivnul_list = zeros(Int32,csc.n)

    mumps_struc = Struc()

    mumps_struc.sym =  2
    mumps_struc.par =  1
    mumps_struc.job = -1
    mumps_struc.comm_fortran = -987654 # MPI.COMM_WORLD.val

    locked_dmumps_c(mumps_struc)
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

    locked_dmumps_c(mumps_struc);
    mumps_struc.info[1] < 0 && throw(SymbolicException())

    csc.nzval.=a

    M = Solver(csc,I,J,sym_perm,pivnul_list,mumps_struc,false,opt,logger)
    finalizer(finalize,M)

    return M
end

function factorize!(M::Solver)
    M.is_singular = false
    M.mumps_struc.job = 2;
    cnt = 0
    while true
        locked_dmumps_c(M.mumps_struc)
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
    locked_dmumps_c(M.mumps_struc)
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
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.mumps_struc.cntl = setindex(M.mumps_struc.cntl,min(M.opt.mumps_pivtolmax,M.mumps_struc.cntl[1]^.5),1)
    @debug(M.logger,"improved quality: pivtol = $(M.mumps_struc.cntl[1])")
    return true
end

function finalize(M::Solver)
    M.mumps_struc.job = -2
    locked_dmumps_c(M.mumps_struc);
end

introduce(::Solver)="mumps"

end # module
