# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module Ma27

import ..MadNLP:
    @kwdef, Logger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, set_options!, SparseMatrixCSC, SubVector, StrideOneVector,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    libhsl, introduce, factorize!, solve!, improve!, is_inertia, inertia, findIJ, nnz

const ma27_default_icntl = Int32[
    6,6,0,2139062143,1,32639,32639,32639,32639,14,9,8,8,9,10,32639,32639,32639,32689,24,11,9,8,9,10,0,0,0,0,0]
const ma27_default_cntl  = [.1,1.0,0.,0.,0.]
const INPUT_MATRIX_TYPE = :csc

@kwdef mutable struct Options <: AbstractOptions
    ma27_pivtol::Float64 = 1e-8
    ma27_pivtolmax::Float64 = 1e-4
    ma27_liw_init_factor::Float64 = 5.
    ma27_la_init_factor::Float64 =5.
    ma27_meminc_factor::Float64 =2.
end

mutable struct Solver <: AbstractLinearSolver
    csc::SparseMatrixCSC{Float64,Int32}
    I::Vector{Int32}
    J::Vector{Int32}

    icntl::Vector{Int32}
    cntl::Vector{Float64}

    info::Vector{Int32}

    a::Vector{Float64}
    a_view::SubVector{Float64}
    la::Int32
    ikeep::Vector{Int32}

    iw::Vector{Int32}
    liw::Int32
    iw1::Vector{Int32}
    nsteps::Vector{Int32}
    w::Vector{Float64}
    maxfrt::Vector{Int32}

    opt::Options
    logger::Logger
end

ma27ad!(n::Cint,nz::Cint,I::StrideOneVector{Cint},J::StrideOneVector{Cint},
        iw::Vector{Cint},liw::Cint,ikeep::Vector{Cint},iw1::Vector{Cint},
        nsteps::Vector{Cint},iflag::Cint,icntl::Vector{Cint},cntl::Vector{Cdouble},
        info::Vector{Cint},ops::Cdouble) = ccall(
            (:ma27ad_,libhsl),
            Nothing,
            (Ref{Cint},Ref{Cint},Ptr{Cint},Ptr{Cint},
             Ptr{Cint},Ref{Cint},Ptr{Cint},Ptr{Cint},
             Ptr{Cint},Ref{Cint},Ptr{Cint},Ptr{Cdouble},
             Ptr{Cint},Ref{Cdouble}),
            n,nz,I,J,iw,liw,ikeep,iw1,nsteps,iflag,icntl,cntl,info,ops)

ma27bd!(n::Cint,nz::Cint,I::StrideOneVector{Cint},J::StrideOneVector{Cint},
        a::StrideOneVector{Cdouble},la::Cint,iw::Vector{Cint},liw::Cint,
        ikeep::Vector{Cint},nsteps::Vector{Cint},maxfrt::Vector{Cint},iw1::Vector{Cint},
        icntl::Vector{Cint},cntl::Vector{Cdouble},info::Vector{Cint}) = ccall(
            (:ma27bd_,libhsl),
            Nothing,
            (Ref{Cint},Ref{Cint},Ptr{Cint},Ptr{Cint},
             Ptr{Cdouble},Ref{Cint},Ptr{Cint},Ref{Cint},
             Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},
             Ptr{Cint},Ptr{Cdouble},Ptr{Cint}),
            n,nz,I,J,a,la,iw,liw,ikeep,nsteps,maxfrt,iw1,icntl,cntl,info)

ma27cd!(n::Cint,a::Vector{Cdouble},la::Cint,iw::Vector{Cint},
        liw::Cint,w::Vector{Cdouble},maxfrt::Vector{Cint},rhs::Vector{Cdouble},
        iw1::Vector{Cint},nsteps::Vector{Cint},icntl::Vector{Cint},
        info::Vector{Cint}) = ccall(
            (:ma27cd_,libhsl),
            Nothing,
            (Ref{Cint},Ptr{Cdouble},Ref{Cint},Ptr{Cint},
             Ref{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cdouble},
             Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
            n,a,la,iw,liw,w,maxfrt,rhs,iw1,nsteps,icntl,info)

function Solver(csc::SparseMatrixCSC;
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Options(),logger=Logger(),kwargs...)

    set_options!(opt,option_dict,kwargs)

    I,J = findIJ(csc)
    nz=Int32(nnz(csc))

    liw= Int32(2*(2*nz+3*csc.n+1))
    iw = Vector{Int32}(undef,liw)
    ikeep= Vector{Int32}(undef,3*csc.n)
    iw1  = Vector{Int32}(undef,2*csc.n)
    nsteps=Int32[1]
    iflag =Int32(0)

    icntl= copy(ma27_default_icntl)
    icntl[1:2] .= 0
    cntl = copy(ma27_default_cntl)
    cntl[1] = opt.ma27_pivtol

    info = Vector{Int32}(undef,20)
    ma27ad!(Int32(csc.n),nz,I,J,iw,liw,ikeep,iw1,nsteps,Int32(0),icntl,cntl,info,0.)
    info[1]<0 && throw(SymbolicException())

    la = ceil(Int32,max(nz,opt.ma27_la_init_factor*info[5]))
    a = Vector{Float64}(undef,la)
    a_view = view(a,1:nnz(csc))
    liw= ceil(Int32,opt.ma27_liw_init_factor*info[6])
    resize!(iw,liw)
    maxfrt=Int32[1]

    return Solver(csc,I,J,icntl,cntl,info,a,a_view,la,ikeep,iw,liw,
                  iw1,nsteps,Vector{Float64}(),maxfrt,opt,logger)
end


function factorize!(M::Solver)
    M.a_view.=M.csc.nzval
    while true
        ma27bd!(Int32(M.csc.n),Int32(nnz(M.csc)),M.I,M.J,M.a,M.la,
                M.iw,M.liw,M.ikeep,M.nsteps,M.maxfrt,
                M.iw1,M.icntl,M.cntl,M.info)
        if M.info[1] == -3
            M.liw = ceil(Int32,M.opt.ma27_meminc_factor*M.liw)
            resize!(M.iw, M.liw)
            @debug(M.logger,"Reallocating memory: liw ($(M.liw))")
        elseif M.info[1] == -4
            M.la = ceil(Int32,M.opt.ma27_meminc_factor*M.la)
            resize!(M.a,M.la)
            @debug(M.logger,"Reallocating memory: la ($(M.la))")
        elseif M.info[1] < 0
            throw(FactorizationException())
        else
            break
        end
    end
    return M
end

function solve!(M::Solver,rhs::StrideOneVector{Float64})
    length(M.w)<M.maxfrt[1] && resize!(M.w,M.maxfrt[1])
    length(M.iw1)<M.nsteps[1] && resize!(M.iw1,M.nsteps[1])
    ma27cd!(Int32(M.csc.n),M.a,M.la,M.iw,M.liw,M.w,M.maxfrt,rhs,
            M.iw1,M.nsteps,M.icntl,M.info)
    M.info[1]<0 && throw(SolveException())
    return rhs
end

is_inertia(::Solver) = true
function inertia(M::Solver)
    rank = M.info[1]==3 ? M.info[2] : rank = M.csc.n
    return (rank-M.info[15],M.csc.n-rank,M.info[15])
end

function improve!(M::Solver)
    if M.cntl[1] == M.opt.ma27_pivtolmax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.cntl[1] = min(M.opt.ma27_pivtolmax,M.cntl[1]^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.cntl[1])")
    return true
end

introduce(::Solver)="ma27"

end # module

# forgiving names
const ma27=Ma27;
const MA27=Ma27;

