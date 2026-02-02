mutable struct LapackCUDASolver{T,MT} <: AbstractLinearSolver{T}
    A::MT
    fact::CuMatrix{T}
    n::Int64
    sol::CuVector{T}
    tau::CuVector{T}
    Λ::CuVector{T}
    work_gpu::CuVector{UInt8}
    lwork_gpu::Csize_t
    work_cpu::Vector{UInt8}
    lwork_cpu::Csize_t
    info::CuVector{Cint}
    ipiv::CuVector{Cint}
    ipiv64::CuVector{Int64}
    opt::LapackOptions
    logger::MadNLPLogger
    legacy::Bool
    params::CuSolverParameters

    function LapackCUDASolver(
        A::MT;
        option_dict::Dict{Symbol,Any} = Dict{Symbol,Any}(),
        opt = LapackOptions(),
        logger = MadNLPLogger(),
        legacy::Bool = true,
        kwargs...,
    ) where {MT<:AbstractMatrix}
        set_options!(opt, option_dict, kwargs...)
        T = eltype(A)
        m,n = size(A)
        @assert m == n
        fact = CuMatrix{T}(undef, m, n)
        sol = CuVector{T}(undef, 0)
        tau = CuVector{T}(undef, 0)
        Λ = CuVector{T}(undef, 0)
        work_gpu = CuVector{UInt8}(undef, 0)
        lwork_gpu = zero(Int64)
        work_cpu = Vector{UInt8}(undef, 0)
        lwork_cpu = zero(Int64)
        info = CuVector{Cint}(undef, 1)
        ipiv = CuVector{Cint}(undef, 0)
        ipiv64 = CuVector{Int64}(undef, 0)
        params = CuSolverParameters()
        solver = new{T,MT}(A, fact, n, sol, tau, Λ, work_gpu, lwork_gpu, work_cpu, lwork_cpu,
                           info, ipiv, ipiv64, opt, logger, legacy, params)
        setup!(solver)
        return solver
    end
end

function setup!(M::LapackCUDASolver)
    if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
        setup_bunchkaufman!(M)
    elseif M.opt.lapack_algorithm == MadNLP.LU
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

function factorize!(M::LapackCUDASolver)
    gpu_transfer!(M.fact, M.A)
    if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
        factorize_bunchkaufman!(M)
    elseif M.opt.lapack_algorithm == MadNLP.LU
        tril_to_full!(M.fact)
        factorize_lu!(M)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        tril_to_full!(M.fact)
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
        function solve!(M::LapackCUDASolver{$T}, x::CuVector{$T})
            if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
                solve_bunchkaufman!(M, x)
            elseif M.opt.lapack_algorithm == MadNLP.LU
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

        is_supported(::Type{LapackCUDASolver}, ::Type{$T}) = true
    end
end

function solve!(M::LapackCUDASolver, x::AbstractVector)
    isempty(M.sol) && resize!(M.sol, M.n)
    copyto!(M.sol, x)
    solve!(M, M.sol)
    copyto!(x, M.sol)
    return x
end

improve!(M::LapackCUDASolver) = false
is_inertia(M::LapackCUDASolver) = (M.opt.lapack_algorithm == MadNLP.CHOLESKY) || (M.opt.lapack_algorithm == MadNLP.EVD)
function inertia(M::LapackCUDASolver)
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

input_type(::Type{LapackCUDASolver}) = :dense
MadNLP.default_options(::Type{LapackCUDASolver}) = LapackOptions()
introduce(M::LapackCUDASolver) = "cuSOLVER v$(CUSOLVER.version()) -- ($(M.opt.lapack_algorithm))"
