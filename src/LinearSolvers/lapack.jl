@kwdef mutable struct LapackOptions <: AbstractOptions
    lapack_algorithm::LinearFactorization = BUNCHKAUFMAN
end

mutable struct LapackCPUSolver{T, MT} <: AbstractLinearSolver{T}
    mat::MT
    fact::Matrix{T}
    work::Vector{T}
    lwork::BlasInt
    info::Ref{BlasInt}
    etc::Dict{Symbol,Any}
    opt::LapackOptions
    logger::MadNLPLogger
end


for (sytrf,sytrs,getrf,getrs,geqrf,ormqr,trsm,potrf,potrs,typ) in (
    (@blasfunc(dsytrf_), @blasfunc(dsytrs_), @blasfunc(dgetrf_), @blasfunc(dgetrs_),
     @blasfunc(dgeqrf_), @blasfunc(dormqr_), @blasfunc(dtrsm_), @blasfunc(dpotrf_), @blasfunc(dpotrs_),
     Float64),
    (@blasfunc(ssytrf_), @blasfunc(ssytrs_), @blasfunc(sgetrf_), @blasfunc(sgetrs_),
     @blasfunc(sgeqrf_), @blasfunc(sormqr_), @blasfunc(strsm_), @blasfunc(spotrf_), @blasfunc(spotrs_),
     Float32)
    )
    @eval begin
        sytrf(uplo,n,a::Matrix{$typ},lda,ipiv,work,lwork,info)=ccall(
            ($(string(sytrf)),libblas),
            Cvoid,
            (Ref{Cchar},Ref{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{BlasInt}),
            uplo,n,a,lda,ipiv,work,lwork,info)
        sytrs(uplo,n,nrhs,a::Matrix{$typ},lda,ipiv,b,ldb,info)=ccall(
            ($(string(sytrs)),libblas),
            Cvoid,
            (Ref{Cchar},Ref{BlasInt},Ref{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{BlasInt}),
            uplo,n,nrhs,a,lda,ipiv,b,ldb,info)
        getrf(m,n,a::Matrix{$typ},lda,ipiv,info)=ccall(
            ($(string(getrf)),libblas),
            Cvoid,
            (Ref{BlasInt},Ref{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{BlasInt},Ptr{BlasInt}),
            m,n,a,lda,ipiv,info)
        getrs(trans,n,nrhs,a::Matrix{$typ},lda,ipiv,b,ldb,info)=ccall(
            ($(string(getrs)),libblas),
            Cvoid,
            (Ref{Cchar},Ref{BlasInt},Ref{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{BlasInt}),
            trans,n,nrhs,a,lda,ipiv,b,ldb,info)
        geqrf(m,n,a::Matrix{$typ},lda,tau,work,lwork,info)=ccall(
            ($(string(geqrf)),libblas),
            Cvoid,
            (Ref{BlasInt},Ref{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{$typ},Ptr{$typ},Ref{BlasInt},Ptr{BlasInt}),
            m,n,a,lda,tau,work,lwork,info)
        ormqr(side,trans,m,n,k,a::Matrix{$typ},lda,tau,c,ldc,work,lwork,info)=ccall(
            ($(string(ormqr)),libblas),
            Cvoid,
            (Ref{Cchar}, Ref{Cchar}, Ref{BlasInt}, Ref{BlasInt},Ref{BlasInt}, Ptr{$typ}, Ref{BlasInt}, Ptr{$typ},Ptr{$typ}, Ref{BlasInt}, Ptr{$typ}, Ref{BlasInt},Ptr{BlasInt}),
            side,trans,m,n,k,a,lda,tau,c,ldc,work,lwork,info)
        trsm(side,uplo,transa,diag,m,n,alpha,a::Matrix{$typ},lda,b,ldb)=ccall(
            ($(string(trsm)),libblas),
            Cvoid,
            (Ref{Cchar},Ref{Cchar},Ref{Cchar},Ref{Cchar},Ref{BlasInt},Ref{BlasInt},Ref{$typ},Ptr{$typ},Ref{BlasInt},Ptr{$typ},Ref{BlasInt}),
            side,uplo,transa,diag,m,n,alpha,a,lda,b,ldb)
        potrf(uplo,n,a::Matrix{$typ},lda,info)=ccall(
            ($(string(potrf)),libblas),
            Cvoid,
            (Ref{Cchar},Ref{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{BlasInt}),
            uplo,n,a,lda,info)
        potrs(uplo,n,nrhs,a::Matrix{$typ},lda,b,ldb,info)=ccall(
            ($(string(potrs)),libblas),
            Cvoid,
            (Ref{Cchar},Ref{BlasInt},Ref{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{$typ},Ref{BlasInt},Ptr{BlasInt}),
            uplo,n,nrhs,a,lda,b,ldb,info)
    end
end

function LapackCPUSolver(
    mat::MT;
    opt=LapackOptions(),
    logger=MadNLPLogger(),
) where {T, MT <: AbstractMatrix{T}}
    fact = Matrix{T}(undef, size(mat))
    etc = Dict{Symbol,Any}()
    work = Vector{T}(undef, 1)
    info = Ref(0)

    return LapackCPUSolver(mat,fact,work,-1,info,etc,opt,logger)
end


function factorize!(M::LapackCPUSolver)
    if M.opt.lapack_algorithm == BUNCHKAUFMAN
        factorize_bunchkaufman!(M)
    elseif M.opt.lapack_algorithm == LU
        factorize_lu!(M)
    elseif M.opt.lapack_algorithm == QR
        factorize_qr!(M)
    elseif M.opt.lapack_algorithm == CHOLESKY
        factorize_cholesky!(M)
    else
        error(LOGGER,"Invalid lapack_algorithm")
    end
end
function solve!(M::LapackCPUSolver{T}, x::Vector{T}) where T
    if M.opt.lapack_algorithm == BUNCHKAUFMAN
        solve_bunchkaufman!(M,x)
    elseif M.opt.lapack_algorithm == LU
        solve_lu!(M,x)
    elseif M.opt.lapack_algorithm == QR
        solve_qr!(M,x)
    elseif M.opt.lapack_algorithm == CHOLESKY
        solve_cholesky!(M,x)
    else
        error(LOGGER,"Invalid lapack_algorithm")
    end
end

function factorize_bunchkaufman!(M::LapackCPUSolver)
    size(M.fact,1) == 0 && return M
    haskey(M.etc,:ipiv) || (M.etc[:ipiv] = Vector{BlasInt}(undef,size(M.mat,1)))
    M.lwork = -1
    M.fact .= M.mat
    sytrf('L',size(M.fact,1),M.fact,size(M.fact,2),M.etc[:ipiv],M.work,M.lwork,M.info)
    M.lwork = BlasInt(real(M.work[1]))
    length(M.work) < M.lwork && resize!(M.work,M.lwork)
    sytrf('L',size(M.fact,1),M.fact,size(M.fact,2),M.etc[:ipiv],M.work,M.lwork,M.info)
    return M
end
function solve_bunchkaufman!(M::LapackCPUSolver,x)
    size(M.fact,1) == 0 && return M
    sytrs('L',size(M.fact,1),1,M.fact,size(M.fact,2),M.etc[:ipiv],x,length(x),M.info)
    return x
end

function factorize_lu!(M::LapackCPUSolver)
    size(M.fact,1) == 0 && return M
    haskey(M.etc,:ipiv) || (M.etc[:ipiv] = Vector{BlasInt}(undef,size(M.mat,1)))
    M.fact .= M.mat
    tril_to_full!(M.fact)
    getrf(size(M.fact,1),size(M.fact,2),M.fact,size(M.fact,2),M.etc[:ipiv],M.info)
    return M
end
function solve_lu!(M::LapackCPUSolver,x)
    size(M.fact,1) == 0 && return M
    getrs('N',size(M.fact,1),1,M.fact,size(M.fact,2),
          M.etc[:ipiv],x,length(x),M.info)
    return x
end


function factorize_qr!(M::LapackCPUSolver{T}) where T
    size(M.fact,1) == 0 && return M
    haskey(M.etc,:tau) || (M.etc[:tau] = Vector{T}(undef,size(M.mat,1)))
    M.lwork = -1
    M.fact .= M.mat
    tril_to_full!(M.fact)
    geqrf(size(M.fact,1),size(M.fact,2),M.fact,size(M.fact,2),M.etc[:tau],M.work,M.lwork,M.info)
    M.lwork = BlasInt(real(M.work[1]))
    length(M.work) < M.lwork && resize!(M.work,M.lwork)
    geqrf(size(M.fact,1),size(M.fact,2),M.fact,size(M.fact,2),M.etc[:tau],M.work,M.lwork,M.info)
    return M
end

function solve_qr!(M::LapackCPUSolver,x)
    size(M.fact,1) == 0 && return M
    M.lwork = -1
    ormqr('L','T',size(M.fact,1),1,length(M.etc[:tau]),M.fact,size(M.fact,2),M.etc[:tau],x,length(x),M.work,M.lwork,M.info)
    M.lwork = BlasInt(real(M.work[1]))
    length(M.work) < M.lwork && resize!(M.work,M.lwork)
    ormqr('L','T',size(M.fact,1),1,length(M.etc[:tau]),M.fact,size(M.fact,2),M.etc[:tau],x,length(x),M.work,M.lwork,M.info)
    trsm('L','U','N','N',size(M.fact,1),1,1.,M.fact,size(M.fact,2),x,length(x))
    return x
end

function factorize_cholesky!(M::LapackCPUSolver)
    size(M.fact,1) == 0 && return M
    M.lwork = -1
    M.fact .= M.mat
    potrf('L',size(M.fact,1),M.fact,size(M.fact,2),M.info)
    return M
end
function solve_cholesky!(M::LapackCPUSolver,x)
    size(M.fact,1) == 0 && return M
    potrs('L',size(M.fact,1),1,M.fact,size(M.fact,2),x,length(x),M.info)
    return x
end

is_inertia(M::LapackCPUSolver) =
    M.opt.lapack_algorithm == BUNCHKAUFMAN || M.opt.lapack_algorithm == CHOLESKY
function inertia(M::LapackCPUSolver)
    if M.opt.lapack_algorithm == BUNCHKAUFMAN
        inertia(M.fact,M.etc[:ipiv],M.info[])
    elseif M.opt.lapack_algorithm == CHOLESKY
        M.info[] == 0 ? (size(M.fact,1),0,0) : (0,size(M.fact,1),0) # later we need to change inertia() to is_inertia_correct() and is_full_rank()
    else
        error(LOGGER,"Invalid lapack_algorithm")
    end
end

function inertia(fact,ipiv,info)
    numneg = num_neg_ev(size(fact,1),fact,ipiv)
    numzero = info > 0 ? 1 : 0
    numpos = size(fact,1) - numneg - numzero
    return (numpos,numzero,numneg)
end

improve!(M::LapackCPUSolver) = false

input_type(::Type{LapackCPUSolver}) = :dense

introduce(M::LapackCPUSolver) = "Lapack-CPU ($(M.opt.lapack_algorithm))"

default_options(::Type{LapackCPUSolver}) = LapackOptions()

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

is_supported(::Type{LapackCPUSolver},::Type{Float32}) = true
is_supported(::Type{LapackCPUSolver},::Type{Float64}) = true
