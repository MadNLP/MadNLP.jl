mutable struct LapackGPUSolver{T} <: AbstractLinearSolver{T}
    A::AbstractMatrix{T}
    fact::CuMatrix{T}
    rhs::CuVector{T}
    work::CuVector{T}
    lwork::Any
    work_host::Vector{T}
    lwork_host::Any
    info::CuVector{Int32}
    etc::Dict{Symbol,Any} # throw some algorithm-specific things here
    opt::LapackOptions
    logger::MadNLPLogger
end

function LapackGPUSolver(
    A::MT;
    option_dict::Dict{Symbol,Any} = Dict{Symbol,Any}(),
    opt = LapackOptions(),
    logger = MadNLPLogger(),
    kwargs...,
) where {T,MT<:AbstractMatrix{T}}
    set_options!(opt, option_dict, kwargs...)
    fact = CuMatrix{T}(undef, size(A))
    rhs = CuVector{T}(undef, size(A, 1))
    work = CuVector{T}(undef, 1)
    lwork = Int32[1]
    work_host = Vector{T}(undef, 1)
    lwork_host = Int32[1]
    info = CuVector{Int32}(undef, 1)
    etc = Dict{Symbol,Any}()

    return LapackGPUSolver{T}(
        A,
        fact,
        rhs,
        work,
        lwork,
        work_host,
        lwork_host,
        info,
        etc,
        opt,
        logger,
    )
end

function factorize!(M::LapackGPUSolver)
    if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
        factorize_bunchkaufman!(M)
    elseif M.opt.lapack_algorithm == MadNLP.LU
        factorize_lu!(M)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        factorize_qr!(M)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        factorize_cholesky!(M)
    else
        error(LOGGER, "Invalid lapack_algorithm")
    end
end
function solve!(M::LapackGPUSolver, x)
    if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
        solve_bunchkaufman!(M, x)
    elseif M.opt.lapack_algorithm == MadNLP.LU
        solve_lu!(M, x)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        solve_qr!(M, x)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        solve_cholesky!(M, x)
    else
        error(LOGGER, "Invalid lapack_algorithm")
    end
end

improve!(M::LapackGPUSolver) = false
introduce(M::LapackGPUSolver) = "Lapack-GPU ($(M.opt.lapack_algorithm))"

for (
    sytrf,
    sytrf_buffer,
    getrf,
    getrf_buffer,
    getrs,
    geqrf,
    geqrf_buffer,
    ormqr,
    ormqr_buffer,
    trsm,
    potrf,
    potrf_buffer,
    potrs,
    sytrs_buffer,
    sytrs,
    typ,
    cutyp,
) in (
    (
        :cusolverDnDsytrf,
        :cusolverDnDsytrf_bufferSize,
        :cusolverDnDgetrf,
        :cusolverDnDgetrf_bufferSize,
        :cusolverDnDgetrs,
        :cusolverDnDgeqrf,
        :cusolverDnDgeqrf_bufferSize,
        :cusolverDnDormqr,
        :cusolverDnDormqr_bufferSize,
        :cublasDtrsm_v2,
        :cusolverDnDpotrf,
        :cusolverDnDpotrf_bufferSize,
        :cusolverDnDpotrs,
        :cusolverDnXsytrs_bufferSize,
        :cusolverDnXsytrs,
        Float64,
        CUDA.R_64F,
    ),
    (
        :cusolverDnSsytrf,
        :cusolverDnSsytrf_bufferSize,
        :cusolverDnSgetrf,
        :cusolverDnSgetrf_bufferSize,
        :cusolverDnSgetrs,
        :cusolverDnSgeqrf,
        :cusolverDnSgeqrf_bufferSize,
        :cusolverDnSormqr,
        :cusolverDnSormqr_bufferSize,
        :cublasStrsm_v2,
        :cusolverDnSpotrf,
        :cusolverDnSpotrf_bufferSize,
        :cusolverDnSpotrs,
        :cusolverDnXsytrs_bufferSize,
        :cusolverDnXsytrs,
        Float32,
        CUDA.R_32F,
    ),
)
    @eval begin
        function factorize_bunchkaufman!(M::LapackGPUSolver{$typ})
            haskey(M.etc, :ipiv) || (M.etc[:ipiv] = CuVector{Int32}(undef, size(M.A, 1)))
            haskey(M.etc, :ipiv64) ||
                (M.etc[:ipiv64] = CuVector{Int64}(undef, length(M.etc[:ipiv])))

            transfer!(M.fact, M.A)
            CUSOLVER.$sytrf_buffer(
                dense_handle(),
                Int32(size(M.fact, 1)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$sytrf(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Int32(size(M.fact, 1)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:ipiv],
                M.work,
                M.lwork[],
                M.info,
            )
            return M
        end

        function solve_bunchkaufman!(M::LapackGPUSolver{$typ}, x)
            copyto!(M.etc[:ipiv64], M.etc[:ipiv])
            copyto!(M.rhs, x)
            CUSOLVER.$sytrs_buffer(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                size(M.fact, 1),
                1,
                $cutyp,
                M.fact,
                size(M.fact, 2),
                M.etc[:ipiv64],
                $cutyp,
                M.rhs,
                length(M.rhs),
                M.lwork,
                M.lwork_host,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            length(M.work_host) < M.lwork_host[] && resize!(work_host, Int(M.lwork_host[]))
            CUSOLVER.$sytrs(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                size(M.fact, 1),
                1,
                $cutyp,
                M.fact,
                size(M.fact, 2),
                M.etc[:ipiv64],
                $cutyp,
                M.rhs,
                length(M.rhs),
                M.work,
                M.lwork[],
                M.work_host,
                M.lwork_host[],
                M.info,
            )
            copyto!(x, M.rhs)

            return x
        end

        function factorize_lu!(M::LapackGPUSolver{$typ})
            haskey(M.etc, :ipiv) || (M.etc[:ipiv] = CuVector{Int32}(undef, size(M.A, 1)))
            transfer!(M.fact, M.A)
            tril_to_full!(M.fact)
            CUSOLVER.$getrf_buffer(
                dense_handle(),
                Int32(size(M.fact, 1)),
                Int32(size(M.fact, 2)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$getrf(
                dense_handle(),
                Int32(size(M.fact, 1)),
                Int32(size(M.fact, 2)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.work,
                M.etc[:ipiv],
                M.info,
            )
            return M
        end

        function solve_lu!(M::LapackGPUSolver{$typ}, x)
            copyto!(M.rhs, x)
            CUSOLVER.$getrs(
                dense_handle(),
                CUBLAS_OP_N,
                Int32(size(M.fact, 1)),
                Int32(1),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:ipiv],
                M.rhs,
                Int32(length(M.rhs)),
                M.info,
            )
            copyto!(x, M.rhs)
            return x
        end

        function factorize_qr!(M::LapackGPUSolver{$typ})
            haskey(M.etc, :tau) || (M.etc[:tau] = CuVector{$typ}(undef, size(M.A, 1)))
            transfer!(M.fact, M.A)
            tril_to_full!(M.fact)
            CUSOLVER.$geqrf_buffer(
                dense_handle(),
                Int32(size(M.fact, 1)),
                Int32(size(M.fact, 2)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$geqrf(
                dense_handle(),
                Int32(size(M.fact, 1)),
                Int32(size(M.fact, 2)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:tau],
                M.work,
                M.lwork[],
                M.info,
            )
            return M
        end

        function solve_qr!(M::LapackGPUSolver{$typ}, x)
            copyto!(M.rhs, x)
            CUSOLVER.$ormqr_buffer(
                dense_handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                Int32(size(M.fact, 1)),
                Int32(1),
                Int32(length(M.etc[:tau])),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:tau],
                M.rhs,
                Int32(length(M.rhs)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$ormqr(
                dense_handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                Int32(size(M.fact, 1)),
                Int32(1),
                Int32(length(M.etc[:tau])),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:tau],
                M.rhs,
                Int32(length(M.rhs)),
                M.work,
                M.lwork[],
                M.info,
            )
            CUBLAS.$trsm(
                handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                Int32(size(M.fact, 1)),
                Int32(1),
                $typ(1),
                M.fact,
                Int32(size(M.fact, 2)),
                M.rhs,
                Int32(length(M.rhs)),
            )
            copyto!(x, M.rhs)
            return x
        end

        function factorize_cholesky!(M::LapackGPUSolver{$typ})
            transfer!(M.fact, M.A)
            CUSOLVER.$potrf_buffer(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Int32(size(M.fact, 1)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$potrf(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Int32(size(M.fact, 1)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.work,
                M.lwork[],
                M.info,
            )
            return M
        end

        function solve_cholesky!(M::LapackGPUSolver{$typ}, x)
            copyto!(M.rhs, x)
            CUSOLVER.$potrs(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Int32(size(M.fact, 1)),
                Int32(1),
                M.fact,
                Int32(size(M.fact, 2)),
                M.rhs,
                Int32(length(M.rhs)),
                M.info,
            )
            copyto!(x, M.rhs)
            return x
        end
    end
end

is_inertia(M::LapackGPUSolver) = M.opt.lapack_algorithm == MadNLP.CHOLESKY  # TODO: implement inertia(M::LapackGPUSolver) for BUNCHKAUFMAN
function inertia(M::LapackGPUSolver)
    if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
        inertia(M.etc[:fact_cpu], M.etc[:ipiv_cpu], M.etc[:info_cpu][])
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        sum(M.info) == 0 ? (size(M.fact, 1), 0, 0) : (0, size(M.fact, 1), 0)
    else
        error(LOGGER, "Invalid lapackcpu_algorithm")
    end
end

input_type(::Type{LapackGPUSolver}) = :dense
MadNLP.default_options(::Type{LapackGPUSolver}) = LapackOptions()
is_supported(::Type{LapackGPUSolver}, ::Type{Float32}) = true
is_supported(::Type{LapackGPUSolver}, ::Type{Float64}) = true

