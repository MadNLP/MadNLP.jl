@enum(MatchingStrategy::Int,COMPLETE=1,COMPLETE2x2=2,CONSTRAINTS=3)

@kwdef mutable struct PardisoMKLOptions <: AbstractOptions
    pardisomkl_num_threads::Int = 1
    pardiso_matching_strategy::MatchingStrategy = COMPLETE2x2
    pardisomkl_max_iterative_refinement_steps::Int = 1
    pardisomkl_msglvl::Int = 0
    pardisomkl_order::Int = 2
end

mutable struct PardisoMKLSolver <: AbstractLinearSolver
    pt::Vector{Ptr{Cvoid}}
    iparm::Vector{Int32}
    perm::Vector{Int32}
    msglvl::Ref{Int32}
    err::Ref{Int32}
    csc::SparseMatrixCSC{Float64,Int32}
    w::Vector{Float64}
    opt::PardisoMKLOptions
    logger::Logger
end

pardisomkl_pardisoinit(pt,mtype::Ref{Cint},iparm::Vector{Cint}) =
    ccall(
        (:pardisoinit,libmkl_rt),
        Cvoid,
        (Ptr{Cvoid},Ptr{Cint},Ptr{Cint}),
        pt,mtype,iparm)
pardisomkl_pardiso(pt,maxfct::Ref{Cint},mnum::Ref{Cint},mtype::Ref{Cint},
                   phase::Ref{Cint},n::Ref{Cint},a::Vector{Cdouble},ia::Vector{Cint},ja::Vector{Cint},
                   perm::Vector{Cint},nrhs::Ref{Cint},iparm::Vector{Cint},msglvl::Ref{Cint},
                   b::Vector{Cdouble},x::Vector{Cdouble},err::Ref{Cint}) =
                       ccall(
                           (:pardiso,libmkl_rt),
                           Cvoid,
                           (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                            Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                            Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble},Ptr{Cint}),
                           pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,x,err)

function pardisomkl_set_num_threads!(n)
    ccall((:mkl_set_dynamic, libmkl_rt),
         Cvoid,
         (Ptr{Cint},),
          Ref{Cint}(0))
    ccall((:mkl_set_num_threads, libmkl_rt),
          Cvoid,
          (Ptr{Cint},),
          Ref{Cint}(n))
end


function PardisoMKLSolver(csc::SparseMatrixCSC;
                opt=PardisoMKLOptions(),logger=Logger(),
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                kwargs...)
    !isempty(kwargs) && (for (key,val) in kwargs; option_dict[key]=val; end)
    set_options!(opt,option_dict)

    w   = Vector{Float64}(undef,csc.n)

    pt = Vector{Ptr{Cvoid}}(undef,64)
    iparm = Vector{Int32}(undef,64)

    perm = Vector{Int32}(undef,csc.n)
    msglvl = Ref{Int32}(0)
    err = Ref{Int32}(0)

    pt.=0
    iparm[1]=0

    pardisomkl_pardisoinit(pt,Ref{Int32}(-2),iparm)

    iparm[1]=1
    iparm[2]=opt.pardisomkl_order # METIS
    iparm[3]=0
    iparm[6]=1
    iparm[8]=opt.pardisomkl_max_iterative_refinement_steps
    iparm[10]=12
    iparm[11]=2
    iparm[13]=1 # matching strateg
    iparm[21]=3 # bunch-kaufman pivotin
    iparm[24]=1 # parallel factorization
    iparm[25]=0 # parallel solv

    pardisomkl_set_num_threads!(opt.pardisomkl_num_threads)
    pardisomkl_pardiso(pt,Ref{Int32}(1),Ref{Int32}(1),Ref{Int32}(-2),Ref{Int32}(11),
                     Ref{Int32}(csc.n),csc.nzval,csc.colptr,csc.rowval,perm,
                       Ref{Int32}(1),iparm,msglvl,Float64[],Float64[],err)
    pardisomkl_set_num_threads!(blas_num_threads[])

    err.x < 0  && throw(SymbolicException())

    M = PardisoMKLSolver(pt,iparm,perm,msglvl,err,csc,w,opt,logger)

    finalizer(finalize,M)

    return M
end
function factorize!(M::PardisoMKLSolver)
    pardisomkl_set_num_threads!(M.opt.pardisomkl_num_threads)
    pardisomkl_pardiso(M.pt,Ref{Int32}(1),Ref{Int32}(1),Ref{Int32}(-2),Ref{Int32}(22),
                     Ref{Int32}(M.csc.n),M.csc.nzval,M.csc.colptr,M.csc.rowval,M.perm,
                       Ref{Int32}(1),M.iparm,M.msglvl,Float64[],Float64[],M.err)
    M.err.x < 0  && throw(FactorizationException())
    pardisomkl_set_num_threads!(blas_num_threads[])

    M.iparm[14] == 0 && return M
    pardisomkl_set_num_threads!(M.opt.pardisomkl_num_threads)
    pardisomkl_pardiso(M.pt,Ref{Int32}(1),Ref{Int32}(1),Ref{Int32}(-2),Ref{Int32}(12),
                     Ref{Int32}(M.csc.n),M.csc.nzval,M.csc.colptr,M.csc.rowval,M.perm,
                       Ref{Int32}(1),M.iparm,M.msglvl,Float64[],Float64[],M.err)
    pardisomkl_set_num_threads!(blas_num_threads[])
    M.err.x < 0  && throw(FactorizationException())
    return M
end
function solve!(M::PardisoMKLSolver,rhs::StrideOneVector{Float64})
    pardisomkl_set_num_threads!(M.opt.pardisomkl_num_threads)
    pardisomkl_pardiso(M.pt,Ref{Int32}(1),Ref{Int32}(1),Ref{Int32}(-2),Ref{Int32}(33),
                     Ref{Int32}(M.csc.n),M.csc.nzval,M.csc.colptr,M.csc.rowval,M.perm,
                       Ref{Int32}(1),M.iparm,M.msglvl,rhs,M.w,M.err)
    pardisomkl_set_num_threads!(blas_num_threads[])
    M.err.x < 0  && throw(SolveException())
    return rhs
end

function finalize(M::PardisoMKLSolver)
    pardisomkl_pardiso(M.pt,Ref{Int32}(1),Ref{Int32}(1),Ref{Int32}(-2),Ref{Int32}(-1),
                     Ref{Int32}(M.csc.n),M.csc.nzval,M.csc.colptr,M.csc.rowval,M.perm,
                     Ref{Int32}(1),M.iparm,M.msglvl,Float64[],M.w,M.err)
end
is_inertia(::PardisoMKLSolver)=true
function inertia(M::PardisoMKLSolver)
    pos = M.iparm[22]
    neg = M.iparm[23]
    return (pos,M.csc.n-pos-neg,neg)
end

function improve!(M::PardisoMKLSolver)
    @debug(M.logger,"improve quality failed.")
    return false
end
introduce(::PardisoMKLSolver)="pardiso-mkl"

input_type(::Type{PardisoMKLSolver}) = :csc
