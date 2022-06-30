# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

const ma57_default_icntl = Int32[0,0,6,1,0,5,1,0,10,0,16,16,10,100,0,0,0,0,0,0]
const ma57_default_cntl  = Float64[1e-8,1.0e-20,0.5,0.0,0.0]

@kwdef mutable struct Ma57Options <: AbstractOptions
    ma57_pivtol::Float64 = 1e-8
    ma57_pivtolmax::Float64 = 1e-4
    ma57_pre_alloc::Float64 = 1.05
    ma57_pivot_order::Int = 5
    ma57_automatic_scaling::Bool =false

    ma57_block_size::Int = 16
    ma57_node_amalgamation::Int = 16
    ma57_small_pivot_flag::Int = 0
end

mutable struct Ma57Solver <: AbstractLinearSolver
    csc::SparseMatrixCSC{Float64,Int32}
    I::Vector{Int32}
    J::Vector{Int32}

    icntl::Vector{Int32}
    cntl::Vector{Float64}

    info::Vector{Int32}
    rinfo::Vector{Float64}

    lkeep::Int32
    keep::Vector{Int32}

    lfact::Int32
    fact::Vector{Float64}

    lifact::Int32
    ifact::Vector{Int32}

    iwork::Vector{Int32}
    lwork::Int32
    work::Vector{Float64}

    opt::Ma57Options
    logger::Logger
end


ma57ad!(n::Cint,nz::Cint,I::Vector{Cint},J::Vector{Cint},lkeep::Cint,
        keep::Vector{Cint},iwork::Vector{Cint},icntl::Vector{Cint},
        info::Vector{Cint},rinfo::Vector{Cdouble}) = ccall(
            (:ma57ad_,libhsl),
            Nothing,
            (Ref{Cint},Ref{Cint},Ptr{Cint},Ptr{Cint},Ref{Cint},
             Ptr{Cint},Ptr{Cint},Ptr{Cint},
             Ptr{Cint},Ptr{Cdouble}),
            n,nz,I,J,lkeep,keep,iwork,icntl,info,rinfo)

ma57bd!(n::Cint,nz::Cint,V::Vector{Cdouble},fact::Vector{Cdouble},
        lfact::Cint,ifact::Vector{Cint},lifact::Cint,lkeep::Cint,
        keep::Vector{Cint},iwork::Vector{Cint},icntl::Vector{Cint},cntl::Vector{Cdouble},
        info::Vector{Cint},rinfo::Vector{Cdouble}) = ccall(
            (:ma57bd_,libhsl),
            Nothing,
            (Ref{Cint},Ref{Cint},Ptr{Cdouble},Ptr{Cdouble},
             Ref{Cint},Ptr{Cint},Ref{Cint},Ref{Cint},
             Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},
             Ptr{Cint},Ptr{Cdouble}),
            n,nz,V,fact,lfact,ifact,lifact,lkeep,keep,iwork,icntl,cntl,info,rinfo)

ma57cd!(job::Cint,n::Cint,fact::Vector{Cdouble},lfact::Cint,
        ifact::Vector{Cint},lifact::Cint,nrhs::Cint,rhs::Vector{Cdouble},
        lrhs::Cint,work::Vector{Cdouble},lwork::Cint,iwork::Vector{Cint},
        icntl::Vector{Cint},info::Vector{Cint}) = ccall(
            (:ma57cd_,libhsl),
            Nothing,
            (Ref{Cint},Ref{Cint},Ptr{Cdouble},Ref{Cint},
             Ptr{Cint},Ref{Cint},Ref{Cint},Ptr{Cdouble},
             Ref{Cint},Ptr{Cdouble},Ref{Cint},Ptr{Cint},
             Ptr{Cint},Ptr{Cint}),
            job,n,fact,lfact,ifact,lifact,nrhs,rhs,lrhs,work,lwork,iwork,icntl,info)

function Ma57Solver(csc::SparseMatrixCSC;
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Ma57Options(),logger=Logger(),kwargs...)

    set_options!(opt,option_dict,kwargs)

    I,J=findIJ(csc)

    icntl= copy(ma57_default_icntl)
    cntl = copy(ma57_default_cntl)

    cntl[1]=opt.ma57_pivtol
    icntl[1]=-1
    icntl[2]=-1
    icntl[3]=-1
    icntl[6]=opt.ma57_pivot_order
    icntl[15]=opt.ma57_automatic_scaling ? 1 : 0
    icntl[11]=opt.ma57_block_size
    icntl[12]=opt.ma57_node_amalgamation

    info = Vector{Int32}(undef,40)
    rinfo= Vector{Float64}(undef,20)

    lkeep=Int32(5*csc.n+nnz(csc)+max(csc.n,nnz(csc))+42)
    keep=Vector{Int32}(undef,lkeep)

    ma57ad!(Int32(csc.n),Int32(nnz(csc)),I,J,lkeep,
            keep,Vector{Int32}(undef,5*csc.n),icntl,
            info,rinfo)

    info[1]<0 && throw(SymbolicException())

    lfact = ceil(Int32,opt.ma57_pre_alloc*info[9])
    lifact = ceil(Int32,opt.ma57_pre_alloc*info[10])

    fact = Vector{Float64}(undef,lfact)
    ifact= Vector{Int32}(undef,lifact)
    iwork= Vector{Int32}(undef,csc.n)
    lwork= csc.n
    work = Vector{Float64}(undef,lwork)

    return Ma57Solver(csc,I,J,icntl,cntl,info,rinfo,lkeep,keep,lfact,fact,lifact,ifact,iwork,lwork,work,opt,logger)
end

function factorize!(M::Ma57Solver)
    while true
        ma57bd!(Int32(M.csc.n),Int32(nnz(M.csc)),M.csc.nzval,M.fact,
                M.lfact,M.ifact,M.lifact,M.lkeep,
                M.keep,M.iwork,M.icntl,M.cntl,
                M.info,M.rinfo)
        if M.info[1] == -3 || M.info[1] == 10
            M.lfact = ceil(Int32,M.opt.ma57_pre_alloc*M.info[17])
            resize!(M.fact, M.lfact)
            @debug(M.logger,"Reallocating memory: lfact ($(M.lfact))")
        elseif M.info[1] == -4 || M.info[1] == 11
            M.lifact = ceil(Int32,M.opt.ma57_pre_alloc*M.info[18])
            resize!(M.ifact,M.lifact)
            @debug(M.logger,"Reallocating memory: lifact ($(M.lifact))")
        elseif M.info[1] < 0
            throw(FactorizationException())
        else
            break
        end
    end
    return M
end

function solve!(M::Ma57Solver,rhs::Vector{Float64})
    ma57cd!(one(Int32),Int32(M.csc.n),M.fact,M.lfact,M.ifact,
            M.lifact,one(Int32),rhs,Int32(M.csc.n),M.work,M.lwork,M.iwork,M.icntl,M.info)
    M.info[1]<0 && throw(SolveException())
    return rhs
end

is_inertia(::Ma57Solver) = true
function inertia(M::Ma57Solver)
    return (M.info[25]-M.info[24],Int32(M.csc.n)-M.info[25],M.info[24])
end
function improve!(M::Ma57Solver)
    if M.cntl[1] == M.opt.ma57_pivtolmax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.cntl[1] = min(M.opt.ma57_pivtolmax,M.cntl[1]^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.cntl[1])")
    return true
end

introduce(::Ma57Solver)="ma57"
input_type(::Type{Ma57Solver}) = :csc
