mutable struct LapackROCSolver{T,MT} <: MadNLP.AbstractLinearSolver{T}
    A::MT
    fact::ROCMatrix{T}
    n::Int64
    sol::ROCVector{T}
    tau::ROCVector{T}
    info::ROCVector{Cint}
    ipiv::ROCVector{Int64}
    alpha::Base.RefValue{T}
    opt::MadNLP.LapackOptions
    logger::MadNLP.MadNLPLogger

    function LapackROCSolver(
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
        info = ROCVector{Cint}(undef, 1)
        ipiv = ROCVector{Int64}(undef, 0)
        alpha = Ref{T}(1)
        solver = new{T,MT}(A, fact, n, sol, tau, info, ipiv, alpha, opt, logger)
        setup!(solver)
        return solver
    end
end

MadNLP.improve!(M::LapackROCSolver) = false
MadNLP.is_inertia(M::LapackROCSolver) = M.opt.lapack_algorithm == MadNLP.CHOLESKY
function MadNLP.inertia(M::LapackROCSolver)
    if M.opt.lapack_algorithm == MadNLP.CHOLESKY
        sum(M.info) == 0 ? (M.n, 0, 0) : (0, M.n, 0)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

MadNLP.input_type(::Type{LapackROCSolver}) = :dense
MadNLP.default_options(::Type{LapackROCSolver}) = MadNLP.LapackOptions(MadNLP.LU)
MadNLP.introduce(M::LapackROCSolver) = "rocSOLVER v$(rocSOLVER.version()) -- ($(M.opt.lapack_algorithm))"

function setup!(M::LapackROCSolver)
    if M.opt.lapack_algorithm == MadNLP.LU
        setup_lu!(M)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        setup_qr!(M)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        setup_cholesky!(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

function MadNLP.factorize!(M::LapackROCSolver)
    MadNLPGPU.gpu_transfer!(M.fact, M.A)
    if M.opt.lapack_algorithm == MadNLP.LU
        MadNLP.tril_to_full!(M.fact)
        factorize_lu!(M)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        MadNLP.tril_to_full!(M.fact)
        factorize_qr!(M)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        factorize_cholesky!(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

for T in (:Float32, :Float64)
    @eval begin
        function MadNLP.solve!(M::LapackROCSolver{$T}, x::ROCVector{$T})
            if M.opt.lapack_algorithm == MadNLP.LU
                solve_lu!(M, x)
            elseif M.opt.lapack_algorithm == MadNLP.QR
                solve_qr!(M, x)
            elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
                solve_cholesky!(M, x)
            else
                error(M.logger, "Invalid lapack_algorithm")
            end
        end

        MadNLP.is_supported(::Type{LapackROCSolver}, ::Type{$T}) = true
    end
end

function MadNLP.solve!(M::LapackROCSolver, x::AbstractVector)
    isempty(M.sol) && resize!(M.sol, M.n)
    copyto!(M.sol, x)
    MadNLP.solve!(M, M.sol)
    copyto!(x, M.sol)
    return x
end

for (potrf, potrs, T) in
    ((:rocsolver_dpotrf_64, :rocsolver_dpotrs_64, :Float64),
     (:rocsolver_spotrf_64, :rocsolver_spotrs_64, :Float32))
    @eval begin
        setup_cholesky!(M::LapackROCSolver{$T}) = M

        function factorize_cholesky!(M::LapackROCSolver{$T})
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

        function solve_cholesky!(M::LapackROCSolver{$T}, x::ROCVector{$T})
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
        function setup_lu!(M::LapackROCSolver{$T})
            resize!(M.ipiv, M.n)
            return M
        end

        function factorize_lu!(M::LapackROCSolver{$T})
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

        function solve_lu!(M::LapackROCSolver{$T}, x::ROCVector{$T})
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

for (geqrf, ormqr, trsm, T) in
    ((:rocsolver_dgeqrf_64, :rocsolver_dormqr, :rocblas_dtrsv_64, :Float64),
     (:rocsolver_sgeqrf_64, :rocsolver_sormqr, :rocblas_strsv_64, :Float32))
    @eval begin
        function setup_qr!(M::LapackROCSolver{$T})
            resize!(M.tau, M.n)
            return M
        end

        function factorize_qr!(M::LapackROCSolver{$T})
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

        function solve_qr!(M::LapackROCSolver{$T}, x::ROCVector{$T})
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
