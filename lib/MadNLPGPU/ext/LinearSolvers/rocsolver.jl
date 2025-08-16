MadNLP.introduce(M::LapackGPUSolver{T,V}) where {T,V<:ROCVector} = "rocSOLVER v$(rocSOLVER.version()) -- ($(M.opt.lapack_algorithm))"

function LapackGPUSolver(
    ::Val{:rocm},
    A::MT;
    option_dict::Dict{Symbol,Any} = Dict{Symbol,Any}(),
    opt = MadNLP.LapackOptions(),
    logger = MadNLP.MadNLPLogger(),
    kwargs...,
) where {MT<:AbstractMatrix}
    T = eltype(A)
    MadNLP.set_options!(opt, option_dict, kwargs...)
    m,n = size(A)
    fact = ROCMatrix{T}(undef, m, n)
    sol = ROCVector{T}(undef, 0)
    tau = ROCVector{T}(undef, 0)
    work_gpu = ROCVector{T}(undef, 0)
    lwork_gpu = zero(Int64)
    work_cpu = Vector{UInt8}(undef, 0)
    lwork_cpu = zero(Int64)
    info = nothing
    info64 = ROCVector{Int64}(undef, 1)
    ipiv = nothing
    ipiv64 = ROCVector{Int64}(undef, 0)
    solver = LapackGPUSolver{T,typeof(tau),typeof(fact),typeof(A),typeof(ipiv),typeof(ipiv64)}(
                A, fact, n, sol, tau, work_gpu, lwork_gpu,
                work_cpu, lwork_cpu, info, info64,
                ipiv, ipiv64, opt, logger)
    setup!(solver)
    return solver
end

for (potrf, potrs, T) in
    ((:rocsolver_dpotrf_64, :rocsolver_dpotrs_64, :Float64),
     (:rocsolver_spotrf_64, :rocsolver_spotrs_64, :Float32))
    @eval begin
        MadNLPGPU.setup_cholesky!(M::LapackGPUSolver{$T,V}) where {V<:ROCVector} = M

        function MadNLPGPU.factorize_cholesky!(M::LapackGPUSolver{$T,V}) where {V<:ROCVector}
            rocSOLVER.$potrf(
                rocBLAS.handle(),
                rocBLAS.rocblas_fill_lower,
                M.n,
                M.fact,
                M.n,
                M.info64,
            )
            return M
        end

        function MadNLPGPU.solve_cholesky!(M::LapackGPUSolver{$T,V}, x::V) where {V<:ROCVector}
            rocSOLVER.$potrs(
                rocBLAS.handle(),
                rocBLAS.rocblas_fill_lower,
                M.n,
                one(Int64),
                M.fact,
                M.n,
                x,
                M.n,
                M.info64,
            )
            return x
        end
    end
end

for (getrf, getrs, T) in
    ((:rocsolver_dgetrf_64, :rocsolver_dgetrs_64, :Float64),
     (:rocsolver_sgetrf_64, :rocsolver_sgetrs_64, :Float32))
    @eval begin
        function MadNLPGPU.setup_lu!(M::LapackGPUSolver{$T,V}) where {V<:ROCVector}
            resize!(M.ipiv64, M.n)
            return M
        end

        function MadNLPGPU.factorize_lu!(M::LapackGPUSolver{$T,V}) where {V<:ROCVector}
            rocSOLVER.$getrf(
                rocBLAS.handle(),
                M.n,
                M.n,
                M.fact,
                M.n,
                M.ipiv64,
                M.info64,
            )
            return M
        end

        function MadNLPGPU.solve_lu!(M::LapackGPUSolver{$T,V}, x::V) where {V<:ROCVector}
            rocSOLVER.$getrs(
                rocBLAS.handle(),
                rocBLAS.rocblas_operation_none,
                M.n,
                one(Int64),
                M.fact,
                M.n,
                M.ipiv64,
                x,
                M.n,
            )
            return x
        end
    end
end

for (geqrf, ormqr, trsm, T) in
    ((:rocsolver_dgeqrf_64, :rocsolver_dormqr, :cublasDtrsm_v2_64, :Float64),
     (:rocsolver_sgeqrf_64, :rocsolver_sormqr, :cublasStrsm_v2_64, :Float32))
    @eval begin
        function MadNLPGPU.setup_qr!(M::LapackGPUSolver{$T,V}) where {V<:ROCVector}
            resize!(M.tau, M.n)
            return M
        end

        function MadNLPGPU.factorize_qr!(M::LapackGPUSolver{$T,V}) where {V<:ROCVector}
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

        function MadNLPGPU.solve_qr!(M::LapackGPUSolver{$T,V}, x::V) where {V<:ROCVector}
            rocSOLVER.$ormqr(
                rocBLAS.handle(),
                rocBLAS.rocblas_side_left,
                rocBLAS.rocblas_operation_transpose,
                Cint(M.n),
                Cint(M.n),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.tau,
                x,
                Cint(M.n),
            )
            rocBLAS.$trsm(
                rocBLAS.handle(),
                rocBLAS.rocblas_side_left,
                rocBLAS.rocblas_fill_upper,
                rocBLAS.rocblas_operation_none,
                rocBLAS.rocblas_diagonal_non_unit,
                M.n,
                one(Int64),
                one($T),
                M.fact,
                M.n,
                x,
                M.n,
            )
            return x
        end
    end
end
