mutable struct LapackROCmSolver{T,MT} <: MadNLP.AbstractLinearSolver{T}
    A::MT
    fact::ROCMatrix{T}
    n::Int64
    sol::ROCVector{T}
    tau::ROCVector{T}
    Λ::ROCVector{T}
    info::ROCVector{Cint}
    ipiv::ROCVector{Int64}
    alpha::Base.RefValue{T}
    beta::Base.RefValue{T}
    opt::MadNLP.LapackOptions
    logger::MadNLP.MadNLPLogger

    function LapackROCmSolver(
        A::MT;
        option_dict::Dict{Symbol,Any} = Dict{Symbol,Any}(),
        opt = MadNLP.LapackOptions(),
        logger = MadNLP.MadNLPLogger(),
        kwargs...,
    ) where {MT<:AbstractMatrix}
        MadNLP.set_options!(opt, option_dict, kwargs...)
        T = eltype(A)
        m,n = size(A)
        @assert m == n
        fact = ROCMatrix{T}(undef, m, n)
        sol = ROCVector{T}(undef, 0)
        tau = ROCVector{T}(undef, 0)
        Λ = ROCVector{T}(undef, 0)
        info = ROCVector{Cint}(undef, 1)
        ipiv = ROCVector{Int64}(undef, 0)
        alpha = Ref{T}(1)
        beta = Ref{T}(0)
        solver = new{T,MT}(A, fact, n, sol, tau, Λ, info, ipiv, alpha, beta, opt, logger)
        setup!(solver)
        return solver
    end
end

MadNLP.improve!(M::LapackROCmSolver) = false
MadNLP.is_inertia(M::LapackROCmSolver) = (M.opt.lapack_algorithm == MadNLP.CHOLESKY) || (M.opt.lapack_algorithm == MadNLP.EVD)
function MadNLP.inertia(M::LapackROCmSolver)
    if M.opt.lapack_algorithm == MadNLP.CHOLESKY
        sum(M.info) == 0 ? (M.n, 0, 0) : (0, M.n, 0)
    elseif M.opt.lapack_algorithm == MadNLP.EVD
        numpos = count(λ -> λ > 0, M.Λ)
        numneg = count(λ -> λ < 0, M.Λ)
        numzero = M.n - numpos - numneg
        (numpos, numzero, numneg)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

MadNLP.input_type(::Type{LapackROCmSolver}) = :dense
MadNLP.default_options(::Type{LapackROCmSolver}) = MadNLP.LapackOptions(MadNLP.EVD)
MadNLP.introduce(M::LapackROCmSolver) = "rocSOLVER v$(rocSOLVER.version()) -- ($(M.opt.lapack_algorithm))"

function setup!(M::LapackROCmSolver)
    if M.opt.lapack_algorithm == MadNLP.LU
        setup_lu!(M)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        setup_qr!(M)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        setup_cholesky!(M)
    elseif M.opt.lapack_algorithm == MadNLP.EVD
        setup_evd!(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

function MadNLP.factorize!(M::LapackROCmSolver)
    MadNLPGPU.gpu_transfer!(M.fact, M.A)
    if M.opt.lapack_algorithm == MadNLP.LU
        MadNLP.tril_to_full!(M.fact)
        factorize_lu!(M)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        MadNLP.tril_to_full!(M.fact)
        factorize_qr!(M)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        factorize_cholesky!(M)
    elseif M.opt.lapack_algorithm == MadNLP.EVD
        factorize_evd!(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

for T in (:Float32, :Float64)
    @eval begin
        function MadNLP.solve_linear_system!(M::LapackROCmSolver{$T}, x::ROCVector{$T})
            if M.opt.lapack_algorithm == MadNLP.LU
                solve_lu!(M, x)
            elseif M.opt.lapack_algorithm == MadNLP.QR
                solve_qr!(M, x)
            elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
                solve_cholesky!(M, x)
            elseif M.opt.lapack_algorithm == MadNLP.EVD
                solve_evd!(M, x)
            else
                error(M.logger, "Invalid lapack_algorithm")
            end
        end

        MadNLP.is_supported(::Type{LapackROCmSolver}, ::Type{$T}) = true
    end
end

function MadNLP.solve_linear_system!(M::LapackROCmSolver, x::AbstractVector)
    isempty(M.sol) && resize!(M.sol, M.n)
    copyto!(M.sol, x)
    MadNLP.solve_linear_system!(M, M.sol)
    copyto!(x, M.sol)
    return x
end

for (potrf, potrs, T) in
    ((:rocsolver_dpotrf_64, :rocsolver_dpotrs_64, :Float64),
     (:rocsolver_spotrf_64, :rocsolver_spotrs_64, :Float32))
    @eval begin
        setup_cholesky!(M::LapackROCmSolver{$T}) = M

        function factorize_cholesky!(M::LapackROCmSolver{$T})
            rocSOLVER.$potrf(
                rocBLAS.handle(),
                rocBLAS.rocblas_fill_lower,
                M.n,
                M.fact,
                M.n,
                M.info,
            )
            return M
        end

        function solve_cholesky!(M::LapackROCmSolver{$T}, x::ROCVector{$T})
            rocSOLVER.$potrs(
                rocBLAS.handle(),
                rocBLAS.rocblas_fill_lower,
                M.n,
                one(Int64),
                M.fact,
                M.n,
                x,
                M.n,
            )
            return x
        end
    end
end

for (getrf, getrs, T) in
    ((:rocsolver_dgetrf_64, :rocsolver_dgetrs_64, :Float64),
     (:rocsolver_sgetrf_64, :rocsolver_sgetrs_64, :Float32))
    @eval begin
        function setup_lu!(M::LapackROCmSolver{$T})
            resize!(M.ipiv, M.n)
            return M
        end

        function factorize_lu!(M::LapackROCmSolver{$T})
            rocSOLVER.$getrf(
                rocBLAS.handle(),
                M.n,
                M.n,
                M.fact,
                M.n,
                M.ipiv,
                M.info,
            )
            return M
        end

        function solve_lu!(M::LapackROCmSolver{$T}, x::ROCVector{$T})
            rocSOLVER.$getrs(
                rocBLAS.handle(),
                rocBLAS.rocblas_operation_none,
                M.n,
                one(Int64),
                M.fact,
                M.n,
                M.ipiv,
                x,
                M.n,
            )
            return x
        end
    end
end

for (geqrf, ormqr, trsv, T) in
    ((:rocsolver_dgeqrf_64, :rocsolver_dormqr, :rocblas_dtrsv_64, :Float64),
     (:rocsolver_sgeqrf_64, :rocsolver_sormqr, :rocblas_strsv_64, :Float32))
    @eval begin
        function setup_qr!(M::LapackROCmSolver{$T})
            resize!(M.tau, M.n)
            return M
        end

        function factorize_qr!(M::LapackROCmSolver{$T})
            rocSOLVER.$geqrf(
                rocBLAS.handle(),
                M.n,
                M.n,
                M.fact,
                M.n,
                M.tau,
            )
            return M
        end

        function solve_qr!(M::LapackROCmSolver{$T}, x::ROCVector{$T})
            rocSOLVER.$ormqr(
                rocBLAS.handle(),
                rocBLAS.rocblas_side_left,
                rocBLAS.rocblas_operation_transpose,
                Cint(M.n),
                Cint(1),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.tau,
                x,
                Cint(M.n),
            )
            rocBLAS.$trsv(
                rocBLAS.handle(),
                rocBLAS.rocblas_fill_upper,
                rocBLAS.rocblas_operation_none,
                rocBLAS.rocblas_diagonal_non_unit,
                M.n,
                M.fact,
                M.n,
                x,
                one(Int64),
            )
            return x
        end
    end
end

for (syevd, gemv, T) in
    ((:rocsolver_dsyevd, :rocblas_dgemv_64, :Float64),
     (:rocsolver_ssyevd, :rocblas_sgemv_64, :Float32))
    @eval begin
        function setup_evd!(M::LapackROCmSolver{$T})
            resize!(M.tau, M.n)
            resize!(M.Λ, M.n)
            return M
        end

        function factorize_evd!(M::LapackROCmSolver{$T})
            rocSOLVER.$syevd(
                rocBLAS.handle(),
                rocSOLVER.rocblas_evect_original,
                rocBLAS.rocblas_fill_lower,
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.Λ,
                M.tau,
                M.info,
            )
            return M
        end

        function solve_evd!(M::LapackROCmSolver{$T}, x::ROCVector{$T})
            rocBLAS.$gemv(
                rocBLAS.handle(),
                rocBLAS.rocblas_operation_transpose,
                M.n,
                M.n,
                M.alpha,
                M.fact,
                M.n,
                x,
                one(Int64),
                M.beta,
                M.tau,
                one(Int64),
            )
            M.tau ./= M.Λ
            rocBLAS.$gemv(
                rocBLAS.handle(),
                rocBLAS.rocblas_operation_none,
                M.n,
                M.n,
                M.alpha,
                M.fact,
                M.n,
                M.tau,
                one(Int64),
                M.beta,
                x,
                one(Int64),
            )
            return x
        end
    end
end
