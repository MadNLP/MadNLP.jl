module MadNLPLapackCPU

import ..MadNLP:
    @kwdef, Logger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, StrideOneVector, set_options!, tril_to_full!,
    libblas, BlasInt, @blasfunc,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia

const INPUT_MATRIX_TYPE = :dense

@enum(Algorithms::Int, BUNCHKAUFMAN = 1, LU = 2, QR =3, CHOLESKY=4)
@kwdef mutable struct Options <: AbstractOptions
    lapackcpu_algorithm::Algorithms = BUNCHKAUFMAN
end

mutable struct Solver <: AbstractLinearSolver
    dense::Matrix{Float64}
    fact::Matrix{Float64}
    work::Vector{Float64}
    lwork::BlasInt
    info::Ref{BlasInt}
    etc::Dict{Symbol,Any}
    opt::Options
    logger::Logger
end

sytrf(uplo,n,a,lda,ipiv,work,lwork,info)=ccall(
    (@blasfunc(dsytrf_),libblas),
    Cvoid,
    (Ref{Cchar},Ref{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{BlasInt}),
    uplo,n,a,lda,ipiv,work,lwork,info)
sytrs(uplo,n,nrhs,a,lda,ipiv,b,ldb,info)=ccall(
    (@blasfunc(dsytrs_),libblas),
    Cvoid,
    (Ref{Cchar},Ref{BlasInt},Ref{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{BlasInt}),
    uplo,n,nrhs,a,lda,ipiv,b,ldb,info)
getrf(m,n,a,lda,ipiv,info)=ccall(
    (@blasfunc(dgetrf_),libblas),
    Cvoid,
    (Ref{BlasInt},Ref{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{BlasInt},Ptr{BlasInt}),
    m,n,a,lda,ipiv,info)
getrs(trans,n,nrhs,a,lda,ipiv,b,ldb,info)=ccall(
    (@blasfunc(dgetrs_),libblas),
    Cvoid,
    (Ref{Cchar},Ref{BlasInt},Ref{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{BlasInt}),
    trans,n,nrhs,a,lda,ipiv,b,ldb,info)
geqrf(m,n,a,lda,tau,work,lwork,info)=ccall(
    (@blasfunc(dgeqrf_),libblas),
    Cvoid,
    (Ref{BlasInt},Ref{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{Cdouble},Ptr{Cdouble},Ref{BlasInt},Ptr{BlasInt}),
    m,n,a,lda,tau,work,lwork,info)
ormqr(side,trans,m,n,k,a,lda,tau,c,ldc,work,lwork,info)=ccall(
    (@blasfunc(dormqr_),libblas),
    Cvoid,
    (Ref{Cchar}, Ref{Cchar}, Ref{BlasInt}, Ref{BlasInt},Ref{BlasInt}, Ptr{Cdouble}, Ref{BlasInt}, Ptr{Cdouble},Ptr{Cdouble}, Ref{BlasInt}, Ptr{Cdouble}, Ref{BlasInt},Ptr{BlasInt}),
    side,trans,m,n,k,a,lda,tau,c,ldc,work,lwork,info)
trsm(side,uplo,transa,diag,m,n,alpha,a,lda,b,ldb)=ccall(
    (@blasfunc(dtrsm_),libblas),
    Cvoid,
    (Ref{Cchar},Ref{Cchar},Ref{Cchar},Ref{Cchar},Ref{BlasInt},Ref{BlasInt},Ref{Cdouble},Ptr{Cdouble},Ref{BlasInt},Ptr{Cdouble},Ref{BlasInt}),
    side,uplo,transa,diag,m,n,alpha,a,lda,b,ldb)
potrf(uplo,n,a,lda,info)=ccall(
    (@blasfunc(dpotrf_),libblas),
    Cvoid,
    (Ref{Cchar},Ref{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{BlasInt}),
    uplo,n,a,lda,info)
potrs(uplo,n,nrhs,a,lda,b,ldb,info)=ccall(
    (@blasfunc(dpotrs_),libblas),
    Cvoid,
    (Ref{Cchar},Ref{BlasInt},Ref{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{Cdouble},Ref{BlasInt},Ptr{BlasInt}),
    uplo,n,nrhs,a,lda,b,ldb,info)

function Solver(dense::Matrix{Float64};
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Options(),logger=Logger(),
                kwargs...)

    set_options!(opt,option_dict,kwargs...)
    fact = copy(dense)

    etc = Dict{Symbol,Any}()
    work = Vector{Float64}(undef, 1)
    info=0

    return Solver(dense,fact,work,-1,info,etc,opt,logger)
end

function factorize!(M::Solver)
    if M.opt.lapackcpu_algorithm == BUNCHKAUFMAN
        factorize_bunchkaufman!(M)
    elseif M.opt.lapackcpu_algorithm == LU
        factorize_lu!(M)
    elseif M.opt.lapackcpu_algorithm == QR
        factorize_qr!(M)
    elseif M.opt.lapackcpu_algorithm == CHOLESKY
        factorize_cholesky!(M)
    else
        error(LOGGER,"Invalid lapackcpu_algorithm")
    end
end
function solve!(M::Solver, x::StrideOneVector{Float64})
    if M.opt.lapackcpu_algorithm == BUNCHKAUFMAN
        solve_bunchkaufman!(M,x)
    elseif M.opt.lapackcpu_algorithm == LU
        solve_lu!(M,x)
    elseif M.opt.lapackcpu_algorithm == QR
        solve_qr!(M,x)
    elseif M.opt.lapackcpu_algorithm == CHOLESKY
        solve_cholesky!(M,x)
    else
        error(LOGGER,"Invalid lapackcpu_algorithm")
    end
end

function factorize_bunchkaufman!(M::Solver)
    size(M.fact,1) == 0 && return M
    haskey(M.etc,:ipiv) || (M.etc[:ipiv] = Vector{BlasInt}(undef,size(M.dense,1)))
    M.lwork = -1
    M.fact .= M.dense
    sytrf('L',size(M.fact,1),M.fact,size(M.fact,2),M.etc[:ipiv],M.work,M.lwork,M.info)
    M.lwork = BlasInt(real(M.work[1]))
    length(M.work) < M.lwork && resize!(M.work,M.lwork)
    sytrf('L',size(M.fact,1),M.fact,size(M.fact,2),M.etc[:ipiv],M.work,M.lwork,M.info)
    return M
end
function solve_bunchkaufman!(M::Solver,x)
    size(M.fact,1) == 0 && return M
    sytrs('L',size(M.fact,1),1,M.fact,size(M.fact,2),M.etc[:ipiv],x,length(x),M.info)
    return x
end

function factorize_lu!(M::Solver)
    size(M.fact,1) == 0 && return M
    haskey(M.etc,:ipiv) || (M.etc[:ipiv] = Vector{BlasInt}(undef,size(M.dense,1)))
    tril_to_full!(M.dense)
    M.fact .= M.dense
    getrf(size(M.fact,1),size(M.fact,2),M.fact,size(M.fact,2),M.etc[:ipiv],M.info)
    return M
end
function solve_lu!(M::Solver,x)
    size(M.fact,1) == 0 && return M
    getrs('N',size(M.fact,1),1,M.fact,size(M.fact,2),
          M.etc[:ipiv],x,length(x),M.info)
    return x
end

function factorize_qr!(M::Solver)
    size(M.fact,1) == 0 && return M
    haskey(M.etc,:tau) || (M.etc[:tau] = Vector{Float64}(undef,size(M.dense,1)))
    tril_to_full!(M.dense)
    M.lwork = -1
    M.fact .= M.dense
    geqrf(size(M.fact,1),size(M.fact,2),M.fact,size(M.fact,2),M.etc[:tau],M.work,M.lwork,M.info)
    M.lwork = BlasInt(real(M.work[1]))
    length(M.work) < M.lwork && resize!(M.work,M.lwork)
    geqrf(size(M.fact,1),size(M.fact,2),M.fact,size(M.fact,2),M.etc[:tau],M.work,M.lwork,M.info)
    return M
end

function solve_qr!(M::Solver,x)
    size(M.fact,1) == 0 && return M
    M.lwork = -1
    ormqr('L','T',size(M.fact,1),1,length(M.etc[:tau]),M.fact,size(M.fact,2),M.etc[:tau],x,length(x),M.work,M.lwork,M.info)
    M.lwork = BlasInt(real(M.work[1]))
    length(M.work) < M.lwork && resize!(M.work,M.lwork)
    ormqr('L','T',size(M.fact,1),1,length(M.etc[:tau]),M.fact,size(M.fact,2),M.etc[:tau],x,length(x),M.work,M.lwork,M.info)
    trsm('L','U','N','N',size(M.fact,1),1,1.,M.fact,size(M.fact,2),x,length(x))
    return x
end

function factorize_cholesky!(M::Solver)
    size(M.fact,1) == 0 && return M
    M.lwork = -1
    M.fact .= M.dense
    potrf('L',size(M.fact,1),M.fact,size(M.fact,2),M.info)
    return M
end
function solve_cholesky!(M::Solver,x)
    size(M.fact,1) == 0 && return M
    potrs('L',size(M.fact,1),1,M.fact,size(M.fact,2),x,length(x),M.info)
    return x
end

is_inertia(M::Solver) =
    M.opt.lapackcpu_algorithm == BUNCHKAUFMAN || M.opt.lapackcpu_algorithm == CHOLESKY
function inertia(M::Solver)
    if M.opt.lapackcpu_algorithm == BUNCHKAUFMAN
        inertia(M.fact,M.etc[:ipiv],M.info[])
    elseif M.opt.lapackcpu_algorithm == CHOLESKY
        M.info[] == 0 ? (size(M.fact,1),0,0) : (0,size(M.fact,1),0) # later we need to change inertia() to is_inertia_correct() and is_full_rank()
    else
        error(LOGGER,"Invalid lapackcpu_algorithm")
    end
end

function inertia(fact,ipiv,info)
    numneg = num_neg_ev(size(fact,1),fact,ipiv)
    numzero = info > 0 ? 1 : 0
    numpos = size(fact,1) - numneg - numzero
    return (numpos,numzero,numneg)
end

improve!(M::Solver) = false

introduce(M::Solver) = "Lapack-CPU ($(M.opt.lapackcpu_algorithm))"

function num_neg_ev(n,D,ipiv)
    numneg = 0
    t = 0
    for k=1:n
        d = D[k,k];
        if ipiv[k] < 0
            if t==0
                t=abs(D[k+1,k])
                d=(d/t)*D[k+1,k+1]-t
            else
                d=t
                t=0
            end
        end
        d<0 && (numneg += 1)
        if d==0
            numneg = -1
            break
        end
    end
    return numneg
end

end # module
