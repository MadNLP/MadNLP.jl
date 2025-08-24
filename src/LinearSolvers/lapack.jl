@kwdef mutable struct LapackOptions <: AbstractOptions
    lapack_algorithm::LinearFactorization = BUNCHKAUFMAN
end

mutable struct LapackCPUSolver{T, MT} <: AbstractLinearSolver{T}
    A::MT
    fact::Matrix{T}
    n::Int64
    sol::Vector{T}
    tau::Vector{T}
    work::Vector{T}
    lwork::BlasInt
    info::Base.RefValue{BlasInt}
    ipiv::Vector{BlasInt}
    opt::LapackOptions
    logger::MadNLPLogger

    function LapackCPUSolver(
        A::MT;
        opt=LapackOptions(),
        logger=MadNLPLogger(),
    ) where {MT <: AbstractMatrix}
        T = eltype(A)
        m,n = size(A)
        @assert m == n
        fact = Matrix{T}(undef, m, n)
        sol = Vector{T}(undef, 0)
        tau = Vector{T}(undef, 0)
        work = Vector{T}(undef, 1)
        lwork = BlasInt(-1)
        info = Ref{BlasInt}(0)
        ipiv = Vector{BlasInt}(undef, 0)
        solver = new{T,MT}(A, fact, n, sol, tau, work, lwork, info, ipiv, opt, logger)
        setup!(solver)
        return solver
    end
end

function setup!(M::LapackCPUSolver)
    if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
        setup_bunchkaufman!(M)
    elseif M.opt.lapack_algorithm == MadNLP.LU
        setup_lu!(M)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        setup_qr!(M)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        setup_cholesky!(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

function factorize!(M::LapackCPUSolver)
    copyto!(M.fact, M.A)
    if M.opt.lapack_algorithm == BUNCHKAUFMAN
        factorize_bunchkaufman!(M)
    elseif M.opt.lapack_algorithm == LU
        tril_to_full!(M.fact)
        factorize_lu!(M)
    elseif M.opt.lapack_algorithm == QR
        tril_to_full!(M.fact)
        factorize_qr!(M)
    elseif M.opt.lapack_algorithm == CHOLESKY
        factorize_cholesky!(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

for T in (:Float32, :Float64)
    @eval begin
        function solve!(M::LapackCPUSolver{$T}, x::Vector{$T})
            if M.opt.lapack_algorithm == BUNCHKAUFMAN
                solve_bunchkaufman!(M, x)
            elseif M.opt.lapack_algorithm == LU
                solve_lu!(M, x)
            elseif M.opt.lapack_algorithm == QR
                solve_qr!(M, x)
            elseif M.opt.lapack_algorithm == CHOLESKY
                solve_cholesky!(M, x)
            else
                error(M.logger, "Invalid lapack_algorithm")
            end
        end

        is_supported(::Type{LapackCPUSolver}, ::Type{$T}) = true
    end
end

function solve!(M::LapackCPUSolver, x::AbstractVector)
    isempty(M.sol) && resize!(M.sol, M.n)
    copyto!(M.sol, x)
    solve!(M, M.sol)
    copyto!(x, M.sol)
    return x
end

for (potrf, potrs, getrf, getrs, sytrf, sytrs, geqrf, ormqr, trsm, T) in
    ((:spotrf_, :spotrs_, :sgetrf_, :sgetrs_, :ssytrf_, :ssytrs_, :sgeqrf_, :sormqr_, :strsm_, :Float32),
     (:dpotrf_, :dpotrs_, :dgetrf_, :dgetrs_, :dsytrf_, :dsytrs_, :dgeqrf_, :dormqr_, :dtrsm_, :Float64))
    @eval begin
        # potrf
        function $potrf(uplo, n, a, lda, info)
            return ccall((@blasfunc($potrf), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong),
                          uplo, n, a, lda, info, 1)
        end

        # potrs
        function $potrs(uplo, n, nrhs, a, lda, b, ldb, info)
            return ccall((@blasfunc($potrs), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T},
                          Ref{BlasInt}, Ref{BlasInt}, Clong),
                          uplo, n, nrhs, a, lda, b, ldb, info, 1)
        end

        # getrf
        function $getrf(m, n, a, lda, ipiv, info)
            return ccall((@blasfunc($getrf), libblastrampoline), Cvoid,
                         (Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
                          m, n, a, lda, ipiv, info)
        end

        # getrs
        function $getrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info)
            return ccall((@blasfunc($getrs), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt},
                          Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong),
                          trans, n, nrhs, a, lda, ipiv, b, ldb, info, 1)
        end

        # sytrf
        function $sytrf(uplo, n, a, lda, ipiv, work, lwork, info)
            return ccall((@blasfunc($sytrf), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$T},
                          Ref{BlasInt}, Ref{BlasInt}, Clong),
                          uplo, n, a, lda, ipiv, work, lwork, info, 1)
        end

        # sytrs
        function $sytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb, info)
            return ccall((@blasfunc($sytrs), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt},
                          Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong), uplo, n, nrhs, a, lda, ipiv,
                          b, ldb, info, 1)
        end

        # geqrf
        function $geqrf(m, n, a, lda, tau, work, lwork, info)
            return ccall((@blasfunc($geqrf), libblastrampoline), Cvoid,
                         (Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ptr{$T},
                          Ref{BlasInt}, Ref{BlasInt}),
                          m, n, a, lda, tau, work, lwork, info)
        end

        # ormqr
        function $ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info)
            return ccall((@blasfunc($ormqr), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T},
                          Ref{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                          Ref{BlasInt}, Clong, Clong),
                          side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1)
        end

        # trsm
        function $trsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
            return ccall((@blasfunc($trsm), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                          Ref{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Clong,
                          Clong, Clong, Clong),
                          side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, 1, 1, 1, 1)
        end

        function setup_cholesky!(M::LapackCPUSolver{$T})
            return M
        end

        function factorize_cholesky!(M::LapackCPUSolver{$T})
            $potrf('L', M.n, M.fact, M.n, M.info)
            return M
        end

        function solve_cholesky!(M::LapackCPUSolver{$T}, x::Vector{$T})
            $potrs('L', M.n, one(BlasInt), M.fact, M.n, x, M.n, M.info)
            return x
        end

        function setup_bunchkaufman!(M::LapackCPUSolver{$T})
            resize!(M.ipiv, M.n)
            $sytrf('L', M.n, M.fact, M.n, M.ipiv, M.work, M.lwork, M.info)
            buffer_size_sytrf = M.work[1] |> BlasInt
            M.lwork = buffer_size_sytrf
            resize!(M.work, M.lwork)
            return M
        end

        function factorize_bunchkaufman!(M::LapackCPUSolver{$T})
            $sytrf('L', M.n, M.fact, M.n, M.ipiv, M.work, M.lwork, M.info)
            return M
        end

        function solve_bunchkaufman!(M::LapackCPUSolver{$T}, x::Vector{$T})
            $sytrs('L', M.n, one(BlasInt), M.fact, M.n, M.ipiv, x, M.n, M.info)
            return x
        end

        function setup_lu!(M::LapackCPUSolver{$T})
            resize!(M.ipiv, M.n)
            return M
        end

        function factorize_lu!(M::LapackCPUSolver{$T})
            $getrf(M.n, M.n, M.fact, M.n, M.ipiv, M.info)
            return M
        end

        function solve_lu!(M::LapackCPUSolver{$T}, x::Vector{$T})
            $getrs('N', M.n, one(BlasInt), M.fact, M.n, M.ipiv, x, M.n, M.info)
            return x
        end

        function setup_qr!(M::LapackCPUSolver{$T})
            resize!(M.tau, M.n)
            $geqrf(M.n, M.n, M.fact, M.n, M.tau, M.work, M.lwork, M.info)
            buffer_size_geqrf = M.work[1] |> BlasInt
            $ormqr('L', 'T', M.n, one(BlasInt), M.n, M.fact, M.n, M.n, M.tau, M.n, M.work, M.lwork, M.info)
            buffer_size_ormqr = M.work[1] |> BlasInt
            M.lwork = max(buffer_size_geqrf, buffer_size_ormqr)
            resize!(M.work, M.lwork)
            return M
        end

        function factorize_qr!(M::LapackCPUSolver{$T})
            $geqrf(M.n, M.n, M.fact, M.n, M.tau, M.work, M.lwork, M.info)
            return M
        end

        function solve_qr!(M::LapackCPUSolver{$T}, x::Vector{$T})
            $ormqr('L', 'T', M.n, one(BlasInt), M.n, M.fact, M.n, M.n, x, M.n, M.work, M.lwork, M.info)
            $trsm('L', 'U', 'N', 'N' , M.n, one(BlasInt), one($T), M.fact, M.n, x, M.n)
            return x
        end
    end
end

improve!(M::LapackCPUSolver) = false
is_inertia(M::LapackCPUSolver) = (M.opt.lapack_algorithm == BUNCHKAUFMAN) || (M.opt.lapack_algorithm == CHOLESKY)
function inertia(M::LapackCPUSolver)
    if M.opt.lapack_algorithm == BUNCHKAUFMAN
        inertia(M.fact, M.ipiv, M.info[])
    elseif M.opt.lapack_algorithm == CHOLESKY
        M.info[] == 0 ? (M.n, 0, 0) : (0, M.n, 0) # later we need to change inertia() to is_inertia_correct() and is_full_rank()
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

input_type(::Type{LapackCPUSolver}) = :dense
default_options(::Type{LapackCPUSolver}) = LapackOptions()
introduce(M::LapackCPUSolver) = "Lapack-CPU ($(M.opt.lapack_algorithm))"

function inertia(fact, ipiv, info)
    numneg = num_neg_ev(size(fact,1), fact, ipiv)
    numzero = info > 0 ? 1 : 0
    numpos = size(fact,1) - numneg - numzero
    return (numpos, numzero, numneg)
end

function num_neg_ev(n, D, ipiv)
    numneg = 0
    t = 0
    for k=1:n
        d = D[k,k]
        if ipiv[k] < 0
            if t == 0
                t = abs(D[k+1,k])
                d = (d/t) * D[k+1,k+1] - t
            else
                d = t
                t = 0
            end
        end
        d < 0 && (numneg += 1)
        if d == 0
            numneg = -1
            break
        end
    end
    return numneg
end
