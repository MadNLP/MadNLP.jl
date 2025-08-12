import CUDSS

@kwdef mutable struct CudssSolverOptions <: MadNLP.AbstractOptions
    # Use LDLᵀ by default in CUDSS as Cholesky can lead to undefined behavior.
    cudss_algorithm::MadNLP.LinearFactorization = MadNLP.LDL
    cudss_ordering::ORDERING = DEFAULT_ORDERING
    cudss_perm::Vector{Cint} = Cint[]
    cudss_ir::Int = 0
    cudss_ir_tol::Float64 = 1e-8
    cudss_pivot_threshold::Float64 = 0.0
    cudss_pivot_epsilon::Float64 = 0.0
    cudss_matching_alg::String = "default"
    cudss_reordering_alg::String = "default"
    cudss_factorization_alg::String = "default"
    cudss_solve_alg::String = "default"
    cudss_matching::Bool = false
    cudss_pivoting::Bool = true
    cudss_hybrid_execute::Bool = false
    cudss_hybrid_memory::Bool = false
    cudss_hybrid_memory_limit::Int = 0
end

function set_cudss_options!(solver, opt::CudssSolverOptions)
    if opt.cudss_ir > 0
        CUDSS.cudss_set(solver, "ir_n_steps", opt.cudss_ir)
        CUDSS.cudss_set(solver, "ir_tol", opt.cudss_ir_tol)
    end
    if opt.cudss_hybrid_memory
        CUDSS.cudss_set(solver, "hybrid_mode", 1)
        if opt.cudss_hybrid_memory_limit > 0
            CUDSS.cudss_set(solver, "hybrid_device_memory_limit", opt.cudss_hybrid_memory_limit)
        end
    end
    if opt.cudss_hybrid_execute
        CUDSS.cudss_set(solver, "hybrid_execute_mode", 1)
    end
    if !opt.cudss_pivoting
        CUDSS.cudss_set(solver, "pivot_type", 'N')
    end
    if opt.cudss_pivot_epsilon > 0.0
        CUDSS.cudss_set(solver, "pivot_epsilon", opt.cudss_pivot_epsilon)
    end
    if opt.cudss_pivot_threshold > 0.0
        CUDSS.cudss_set(solver, "pivot_threshold", opt.cudss_pivot_threshold)
    end
    if opt.cudss_matching
        CUDSS.cudss_set(solver, "use_matching", 1)
        if opt.cudss_matching_alg != "default"
            CUDSS.cudss_set(solver, "matching_alg", opt.cudss_matching_alg)
        end
    end
    if opt.cudss_reordering_alg != "default"
        CUDSS.cudss_set(solver, "reordering_alg", opt.cudss_reordering_alg)
    end
    if opt.cudss_factorization_alg != "default"
        CUDSS.cudss_set(solver, "factorization_alg", opt.cudss_factorization_alg)
    end
    if opt.cudss_solve_alg != "default"
        CUDSS.cudss_set(solver, "solve_alg", opt.cudss_solve_alg)
    end
end

mutable struct CUDSSSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::CUDSS.CudssSolver{T}
    tril::CUSPARSE.CuSparseMatrixCSC{T}
    x_gpu::CUDSS.CudssMatrix{T}
    b_gpu::CUDSS.CudssMatrix{T}
    fresh_factorization::Bool

    opt::CudssSolverOptions
    logger::MadNLP.MadNLPLogger
end

function CUDSSSolver(
    csc::CUSPARSE.CuSparseMatrixCSC{T};
    opt=CudssSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
    ) where T
    n, m = size(csc)
    @assert n == m

    view = 'U'
    structure = 'G'
    # We need view = 'F' for the sparse LU decomposition
    (opt.cudss_algorithm == MadNLP.LU) && error("The sparse LU of cuDSS is not supported.")
    (opt.cudss_algorithm == MadNLP.CHOLESKY) && (structure = "SPD")
    (opt.cudss_algorithm == MadNLP.LDL) && (structure = "S")

    matrix = CUSPARSE.CuSparseMatrixCSR(csc.colPtr, csc.rowVal, csc.nzVal, csc.dims)
    solver = CUDSS.CudssSolver(matrix, structure, view)
    set_cudss_options!(solver, opt)

    if opt.cudss_ordering != DEFAULT_ORDERING
        if opt.cudss_ordering == METIS_ORDERING
            A = SparseMatrixCSC(csc)
            A = A + A' - LinearAlgebra.Diagonal(A)
            G = Metis.graph(A, check_hermitian=false)
            opt.cudss_perm, _ = Metis.permutation(G)
        elseif opt.cudss_ordering == AMD_ORDERING
            A = SparseMatrixCSC(csc)
            opt.cudss_perm = AMD.amd(A)
        elseif opt.cudss_ordering == SYMAMD_ORDERING
            A = SparseMatrixCSC(csc)
            opt.cudss_perm = AMD.symamd(A)
        elseif opt.cudss_ordering == COLAMD_ORDERING
            A = SparseMatrixCSC(csc)
            opt.cudss_perm = AMD.colamd(A)
        elseif opt.cudss_ordering == USER_ORDERING
            (!isempty(opt.cudss_perm) && isperm(opt.cudss_perm)) || error("The vector opt.cudss_perm is not a valid permutation.")
        else
            error("The ordering $(opt.cudss_ordering) is not supported.")
        end
        CUDSS.cudss_set(solver, "user_perm", opt.cudss_perm)
    end

    # The phase "analysis" is "reordering" combined with "symbolic_factorization"
    x_gpu = CUDSS.CudssMatrix(T, n)
    b_gpu = CUDSS.CudssMatrix(T, n)
    CUDSS.cudss("analysis", solver, x_gpu, b_gpu)

    return CUDSSSolver(
        solver, csc,
        x_gpu, b_gpu,
        true, opt, logger,
    )
end

function MadNLP.factorize!(M::CUDSSSolver)
    CUDSS.cudss_set(M.inner.matrix, nonzeros(M.tril))
    if M.fresh_factorization
        CUDSS.cudss("factorization", M.inner, M.x_gpu, M.b_gpu)
        M.fresh_factorization = false
    else
        CUDSS.cudss("refactorization", M.inner, M.x_gpu, M.b_gpu)
    end
    CUDA.synchronize()
    return M
end

function MadNLP.solve!(M::CUDSSSolver{T}, xb::CuVector{T}) where T
    CUDSS.cudss_set(M.b_gpu, xb)
    CUDSS.cudss_set(M.x_gpu, xb)
    CUDSS.cudss("solve", M.inner, M.x_gpu, M.b_gpu)
    CUDA.synchronize()
    return xb
end

MadNLP.input_type(::Type{CUDSSSolver}) = :csc
MadNLP.default_options(::Type{CUDSSSolver}) = CudssSolverOptions()
MadNLP.is_inertia(M::CUDSSSolver) = true  # Uncomment if MadNLP.LU is supported -- (M.opt.cudss_algorithm ∈ (MadNLP.CHOLESKY, MadNLP.LDL))
function inertia(M::CUDSSSolver)
    n = size(M.tril, 1)
    if M.opt.cudss_algorithm == MadNLP.CHOLESKY
        info = CUDSS.cudss_get(M.inner, "info")
        if info == 0
            return (n, 0, 0)
        else
            CUDSS.cudss_set(M.inner, "info", 0)
            return (n-2, 1, 1)
        end
    elseif M.opt.cudss_algorithm == MadNLP.LDL
        # N.B.: cuDSS does not always return the correct inertia.
        (k, l) = CUDSS.cudss_get(M.inner, "inertia")
        k = min(n, k) # TODO: add safeguard for inertia
        return (k, n - k - l, l)
    end
end
MadNLP.improve!(M::CUDSSSolver) = false
MadNLP.is_supported(::Type{CUDSSSolver},::Type{Float32}) = true
MadNLP.is_supported(::Type{CUDSSSolver},::Type{Float64}) = true
MadNLP.introduce(M::CUDSSSolver) = "cuDSS v$(CUDSS.version())"
