mutable struct LapackOneMKLSolver{T,MT} <: MadNLP.AbstractLinearSolver{T}
    A::MT
    fact::oneMatrix{T}
    n::Int64
    sol::oneVector{T}
    tau::oneVector{T}
    Λ::oneVector{T}
    info::oneVector{Cint}
    ipiv::oneVector{Int64}
    scratchpad::oneVector{T}
    scratchpad_size::Int64
    device_queue::SYCL.syclQueue_t
    alpha::Base.RefValue{T}
    beta::Base.RefValue{T}
    opt::MadNLP.LapackOptions
    logger::MadNLP.MadNLPLogger

    function LapackOneMKLSolver(
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
        fact = oneMatrix{T}(undef, m, n)
        sol = oneVector{T}(undef, 0)
        tau = oneVector{T}(undef, 0)
        Λ = oneVector{T}(undef, 0)
        info = oneVector{Cint}(undef, 1)
        ipiv = oneVector{Int64}(undef, 0)
        scratchpad = oneVector{T}(undef, 0)
        scratchpad_size = 0
        # Get the device queue from the oneAPI context
        queue = oneAPI.global_queue(oneAPI.context(fact), oneAPI.device(fact))
        device_queue = oneAPI.sycl_queue(queue)
        alpha = Ref{T}(1)
        beta = Ref{T}(0)
        solver = new{T,MT}(A, fact, n, sol, tau, Λ, info, ipiv, scratchpad, scratchpad_size, device_queue, alpha, beta, opt, logger)
        setup!(solver)
        return solver
    end
end

MadNLP.improve!(M::LapackOneMKLSolver) = false
MadNLP.is_inertia(M::LapackOneMKLSolver) = (M.opt.lapack_algorithm == MadNLP.CHOLESKY) || (M.opt.lapack_algorithm == MadNLP.EVD)
function MadNLP.inertia(M::LapackOneMKLSolver)
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

MadNLP.input_type(::Type{LapackOneMKLSolver}) = :dense
MadNLP.default_options(::Type{LapackOneMKLSolver}) = MadNLP.LapackOptions(MadNLP.EVD)
MadNLP.introduce(M::LapackOneMKLSolver) = "OneAPI -- ($(M.opt.lapack_algorithm))"
# MadNLP.introduce(M::LapackOneMKLSolver) = "OneAPI v$(oneAPI.version()) -- ($(M.opt.lapack_algorithm))"

function setup!(M::LapackOneMKLSolver)
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

function MadNLP.factorize!(M::LapackOneMKLSolver)
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
        function MadNLP.solve!(M::LapackOneMKLSolver{$T}, x::oneVector{$T})
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

        MadNLP.is_supported(::Type{LapackOneMKLSolver}, ::Type{$T}) = true
    end
end

function MadNLP.solve!(M::LapackOneMKLSolver, x::AbstractVector)
    isempty(M.sol) && resize!(M.sol, M.n)
    copyto!(M.sol, x)
    MadNLP.solve!(M, M.sol)
    copyto!(x, M.sol)
    return x
end

for (potrf, potrf_buffer, potrs, potrs_buffer, T) in
    ((:onemklDpotrf, :onemklDpotrf_scratchpad_size, :onemklDpotrs, :onemklDpotrs_scratchpad_size, :Float64),
     (:onemklSpotrf, :onemklSpotrf_scratchpad_size, :onemklSpotrs, :onemklSpotrs_scratchpad_size, :Float32))
    @eval begin
        function setup_cholesky!(M::LapackOneMKLSolver{$T})
            potrf_scratchpad_size = Support.$potrf_buffer(M.device_queue, 'L', M.n, M.n)
            potrs_scratchpad_size = Support.$potrs_buffer(M.device_queue, 'L', M.n, one(Int64), M.n, M.n)
            M.scratchpad_size = max(potrf_scratchpad_size, potrs_scratchpad_size)
            resize!(M.scratchpad, M.scratchpad_size)
            return M
        end

        function factorize_cholesky!(M::LapackOneMKLSolver{$T})
            Support.$potrf(
                M.device_queue,
                'L',
                M.n,
                M.fact,
                M.n,
                M.scratchpad,
                M.scratchpad_size,
            )
            return M
        end

        function solve_cholesky!(M::LapackOneMKLSolver{$T}, x::oneVector{$T})
            Support.$potrs(
                M.device_queue,
                'L',
                n,
                one(Int64),
                M.fact,
                M.n,
                x,
                M.n,
                M.scratchpad,
                M.scratchpad_size,
            )
            return x
        end
    end
end

for (getrf, getrf_buffer, getrs, getrs_buffer, T) in
    ((:onemklDgetrf, :onemklDgetrf_scratchpad_size, :onemklDgetrs, :onemklDgetrs_scratchpad_size, :Float64),
     (:onemklSgetrf, :onemklSgetrf_scratchpad_size, :onemklSgetrs, :onemklSgetrs_scratchpad_size, :Float32))
    @eval begin
        function setup_lu!(M::LapackOneMKLSolver{$T})
            resize!(M.ipiv, M.n)
            getrf_scratchpad_size = Support.$getrf_buffer(M.device_queue, M.n, M.n, M.n)
            getrs_scratchpad_size = Support.$getrs_buffer(M.device_queue, 'N', M.n, one(Int64), M.n, M.n)
            M.scratchpad_size = max(getrf_scratchpad_size, getrs_scratchpad_size)
            resize!(M.scratchpad, M.scratchpad_size)
            return M
        end

        function factorize_lu!(M::LapackOneMKLSolver{$T})
            Support.$getrf(
                M.device_queue,
                M.n,
                M.n,
                M.fact,
                M.n,
                M.ipiv,
                M.scratchpad,
                M.scratchpad_size,
            )
            return M
        end

        function solve_lu!(M::LapackOneMKLSolver{$T}, x::oneVector{$T})
            Support.$getrs(
                M.device_queue,
                'N',
                M.n,
                one(Int64),
                M.fact,
                M.n,
                M.ipiv,
                x,
                M.n,
                M.scratchpad,
                M.scratchpad_size,
            )
            return x
        end
    end
end

for (geqrf, geqrf_buffer, ormqr, ormqr_buffer, trsv, T) in
    ((:onemklDgeqrf, :onemklDgeqrf_scratchpad_size, :onemklDormqr, :onemklDormqr_scratchpad_size, :onemklDtrsv, :Float64),
     (:onemklSgeqrf, :onemklSgeqrf_scratchpad_size, :onemklSormqr, :onemklSormqr_scratchpad_size, :onemklStrsv, :Float32))
    @eval begin
        function setup_qr!(M::LapackOneMKLSolver{$T})
            resize!(M.tau, M.n)
            geqrf_scratchpad_size = Support.$geqrf_buffer(M.device_queue, M.n, M.n, M.n)
            # TODO: Fix ormqr buffer size calculation - undefined variables
            # ormqr_scratchpad_size = Support.$ormqr_buffer(M.device_queue, side, trans, m, n, k, lda, ldc)
            M.scratchpad_size = geqrf_scratchpad_size
            resize!(M.scratchpad, M.scratchpad_size)
            return M
        end

        function factorize_qr!(M::LapackOneMKLSolver{$T})
            Support.$geqrf(
                M.device_queue,
                M.n,
                M.n,
                M.fact,
                M.n,
                M.tau,
                M.scratchpad,
                M.scratchpad_size,
            )
            return M
        end

        function solve_qr!(M::LapackOneMKLSolver{$T}, x::oneVector{$T})
            Support.$ormqr(
                M.device_queue,
                M.side,
                M.trans,
                M.n,
                M.n,
                M.n,
                M.fact,
                M.n,
                M.tau,
                c,
                ldc,
                M.scratchpad,
                M.scratchpad_size,
            )
            oneBLAS.$trsv(
                oneBLAS.handle(),
                oneBLAS.oneblas_side_left,
                oneBLAS.oneblas_fill_upper,
                oneBLAS.oneblas_operation_none,
                oneBLAS.oneblas_diagonal_non_unit,
                M.n,
                one(Int64),
                M.alpha,
                M.fact,
                M.n,
                x,
                M.n,
            )
            return x
        end
    end
end

for (syevd, syevd_buffer, gemv, T) in
    ((:onemklDsyevd, :onemklDsyevd_scratchpad_size, :onemklDgemv, :Float64),
     (:onemklSsyevd, :onemklSsyevd_scratchpad_size, :onemklSgemv, :Float32))
    @eval begin
        function setup_evd!(M::LapackOneMKLSolver{$T})
            resize!(M.tau, M.n)
            resize!(M.Λ, M.n)
            M.scratchpad_size = Support.$syevd_buffer(M.device_queue, 'V', 'L', M.n, M.n)
            resize!(M.scratchpad, M.scratchpad_size)
            return M
        end

        function factorize_evd!(M::LapackOneMKLSolver{$T})
            Support.$syevd(
                M.device_queue,
                'V',
                'L',
                M.n,
                M.fact,
                M.n,
                M.Λ,
                M.scratchpad,
                M.scratchpad_size,
            )
            return M
        end

        function solve_evd!(M::LapackOneMKLSolver{$T}, x::oneVector{$T})
            Support.$gemv(
                M.device_queue,
                'T',
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
            Support.$gemv(
                M.device_queue,
                'N',
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
