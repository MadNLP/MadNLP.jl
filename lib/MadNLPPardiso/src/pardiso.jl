@enum(MatchingStrategy::Int,COMPLETE=1,COMPLETE2x2=2,CONSTRAINTS=3)

@kwdef mutable struct PardisoOptions <: AbstractOptions
    pardiso_matching_strategy::MatchingStrategy = COMPLETE2x2
    pardiso_max_inner_refinement_steps::Int = 1
    pardiso_msglvl::Int = 0
    pardiso_order::Int = 2
    pardiso_log_level::String = ""
end

mutable struct PardisoSolver{T} <: AbstractLinearSolver{T}
    pt::Vector{Int}
    iparm::Vector{Int32}
    dparm::Vector{T}
    perm::Vector{Int32}
    msglvl::Ref{Int32}
    err::Ref{Int32}
    csc::SparseMatrixCSC{T,Int32}
    w::Vector{T}
    opt::PardisoOptions
    logger::MadNLPLogger
end

_pardisoinit(
    pt::Vector{Int},mtype::Ref{Cint},solver::Ref{Cint},iparm::Vector{Cint},
    dparm::Vector{Cdouble},err::Ref{Cint}) =
        ccall(
            (:pardisoinit,libpardiso),
            Cvoid,
            (Ptr{Int},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint}),
            pt,mtype,solver,iparm,dparm,err)
_pardiso(
    pt::Vector{Int},maxfct::Ref{Cint},mnum::Ref{Cint},mtype::Ref{Cint},
    phase::Ref{Cint},n::Ref{Cint},a::Vector{Cdouble},ia::Vector{Cint},ja::Vector{Cint},
    perm::Vector{Cint},nrhs::Ref{Cint},iparm::Vector{Cint},msglvl::Ref{Cint},
    b::Vector{Cdouble},x::Vector{Cdouble},err::Ref{Cint},dparm::Vector{Cdouble}) =
        ccall(
            (:pardiso,libpardiso),
            Cvoid,
            (Ptr{Int},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cint},
             Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cint},Ptr{Cdouble}),
            pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,x,err,dparm)

function PardisoSolver(csc::SparseMatrixCSC{T,Int32};
                opt=PardisoOptions(),logger=MadNLPLogger(),
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                kwargs...) where T
    !isempty(kwargs) && (for (key,val) in kwargs; option_dict[key]=val; end)
    set_options!(opt,option_dict)

    w   = Vector{T}(undef,csc.n)

    pt = Vector{Int}(undef,64)
    iparm = Vector{Int32}(undef,64)
    dparm = Vector{T}(undef,64)
    perm = Vector{Int32}(undef,csc.n)
    msglvl = Ref{Int32}(0)
    err = Ref{Int32}(0)

    pt.=0
    iparm[1]=0

    _pardisoinit(pt,Ref{Int32}(-2),Ref{Int32}(0),iparm,dparm,err)
    err.x < 0  && throw(SymbolicException())

    iparm[1]=1
    iparm[2]=opt.pardiso_order # METIS
    iparm[3]=haskey(ENV,"OMP_NUM_THREADS") ? parse(Int32,ENV["OMP_NUM_THREADS"]) : 1
    iparm[6]=1
    iparm[8]=opt.pardiso_max_inner_refinement_steps
    iparm[10]=12 # pivot perturbation
    iparm[11]=0 # disable scaling
    iparm[13]=1 # matching strategy
    iparm[21]=3 # bunch-kaufman pivotin
    iparm[24]=1 # parallel factorization
    iparm[25]=1 # parallel solv
    iparm[29]= T == Float64 ? 0 : 1

    _pardiso(pt,Ref{Int32}(1),Ref{Int32}(1),Ref{Int32}(-2),Ref{Int32}(11),
             Ref{Int32}(csc.n),csc.nzval,csc.colptr,csc.rowval,perm,
             Ref{Int32}(1),iparm,msglvl,T[],T[],err,dparm)
    err.x < 0  && throw(SymbolicException())

    M = PardisoSolver{T}(pt,iparm,dparm,perm,msglvl,err,csc,w,opt,logger)

    finalizer(finalize,M)

    return M
end
function factorize!(M::PardisoSolver{T}) where T
    _pardiso(M.pt,Ref{Int32}(1),Ref{Int32}(1),Ref{Int32}(-2),Ref{Int32}(22),
             Ref{Int32}(M.csc.n),M.csc.nzval,M.csc.colptr,M.csc.rowval,M.perm,
             Ref{Int32}(1),M.iparm,M.msglvl,T[],T[],M.err,M.dparm)
    M.err.x < 0  && throw(FactorizationException())
    M.iparm[14] == 0 && return M
    _pardiso(M.pt,Ref{Int32}(1),Ref{Int32}(1),Ref{Int32}(-2),Ref{Int32}(12),
             Ref{Int32}(M.csc.n),M.csc.nzval,M.csc.colptr,M.csc.rowval,M.perm,
             Ref{Int32}(1),M.iparm,M.msglvl,T[],T[],M.err,M.dparm)
    M.err.x < 0  && throw(FactorizationException())
    return M
end
function solve!(M::PardisoSolver{T},rhs::Vector{T}) where T
    _pardiso(M.pt,Ref{Int32}(1),Ref{Int32}(1),Ref{Int32}(-2),Ref{Int32}(33),
             Ref{Int32}(M.csc.n),M.csc.nzval,M.csc.colptr,M.csc.rowval,M.perm,
             Ref{Int32}(1),M.iparm,M.msglvl,rhs,M.w,M.err,M.dparm)
    M.err.x < 0  && throw(SolveException())
    return rhs
end
function finalize(M::PardisoSolver{T}) where T
    _pardiso(M.pt,Ref{Int32}(1),Ref{Int32}(1),Ref{Int32}(-2),Ref{Int32}(-1),
             Ref{Int32}(M.csc.n),M.csc.nzval,M.csc.colptr,M.csc.rowval,M.perm,
             Ref{Int32}(1),M.iparm,M.msglvl,T[],M.w,M.err,M.dparm)
end
is_inertia(::PardisoSolver)=true
function inertia(M::PardisoSolver)
    pos = M.iparm[22]
    neg = M.iparm[23]
    return (pos,M.csc.n-pos-neg,neg)
end
function improve!(M::PardisoSolver)
    @debug(M.logger,"improve quality failed.")
    return false
end

introduce(::PardisoSolver)="pardiso"
input_type(::Type{PardisoSolver}) = :csc
default_options(::Type{PardisoSolver}) = PardisoOptions()
is_supported(::Type{PardisoSolver},::Type{Float32}) = true
is_supported(::Type{PardisoSolver},::Type{Float64}) = true
