import CUDSS

@kwdef mutable struct CudssSolverOptions <: MadNLP.AbstractOptions
    # Use LDLᵀ by default in CUDSS as Cholesky can lead to undefined behavior.
    cudss_algorithm::MadNLP.LinearFactorization = MadNLP.LDL
    ordering::ORDERING = DEFAULT_ORDERING
    perm::Vector{Cint} = Cint[]
    ir::Int = 0
    hybrid::Bool = false
    pivoting::Bool = true
end

mutable struct CUDSSSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing, CUDSS.CudssSolver}
    tril
    tril_buffer
    x_gpu::CUDA.CuVector{T}
    b_gpu::CUDA.CuVector{T}

    opt::CudssSolverOptions
    logger::MadNLP.MadNLPLogger
end

function CUDSSSolver(
    csc;
    opt=CudssSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
    )
    T = eltype(csc.nzval)
    n, m = size(csc)
    @assert n == m

    view = 'U'
    structure = if opt.cudss_algorithm == MadNLP.LU
        "G"
    elseif opt.cudss_algorithm == MadNLP.CHOLESKY
        "SPD"
    elseif opt.cudss_algorithm == MadNLP.LDL
        "S"
    end

    matrix = CUDSS.CudssMatrix(
        CUSPARSE.CuSparseMatrixCSR(CUDA.CuArray(csc.colptr), CUDA.CuArray(csc.rowval), CUDA.CuArray(csc.nzval), (csc.n, csc.n)),
        structure,
        view
    )

    # TODO: pass config options here.
    config = CUDSS.CudssConfig()
    data = CUDSS.CudssData()
    solver = CUDSS.CudssSolver(matrix, config, data)

    if opt.ordering != DEFAULT_ORDERING
        if opt.ordering == METIS_ORDERING
            A = SparseMatrixCSC(csc)
            A = A + A' - LinearAlgebra.Diagonal(A)
            G = Metis.graph(A, check_hermitian=false)
            opt.perm, _ = Metis.permutation(G)
        elseif opt.ordering == AMD_ORDERING
            A = SparseMatrixCSC(csc)
            opt.perm = AMD.amd(A)
        elseif opt.ordering == USER_ORDERING
            (!isempty(opt.perm) && isperm(opt.perm)) || error("The vector opt.perm is not a valid permutation.")
        else
            error("The ordering $(opt.ordering) is not supported.")
        end
        CUDSS.cudss_set(solver, "user_perm", opt.perm)
    end
    (opt.ir > 0) && CUDSS.cudss_set(solver, "ir_n_steps", opt.ir)
    opt.hybrid && CUDSS.cudss_set(solver, "hybrid_mode", 1)
    !opt.pivoting && CUDSS.cudss_set(solver, "pivot_type", 'N')

    x_gpu = CUDA.zeros(T, n)
    b_gpu = CUDA.zeros(T, n)

    CUDSS.cudss("analysis", solver, x_gpu, b_gpu)

    return CUDSSSolver(
        solver, csc, CuArray(nonzeros(csc)),
        # full, tril_to_full_view,
        x_gpu, b_gpu,
        opt, logger
    )
end

function MadNLP.factorize!(M::CUDSSSolver)
    # copyto!(M.full.nzVal, M.tril_to_full_view)
    copyto!(M.tril_buffer, nonzeros(M.tril))
    CUDSS.cudss_set(M.inner.matrix, M.tril_buffer)
    CUDSS.cudss("factorization", M.inner, M.x_gpu, M.b_gpu)
    synchronize(CUDABackend())
    return M
end

function MadNLP.solve!(M::CUDSSSolver{T}, x) where T
    copyto!(M.b_gpu, x)
    CUDSS.cudss("solve", M.inner, M.x_gpu, M.b_gpu)
    synchronize(CUDABackend())
    copyto!(x, M.x_gpu)
    return x
end

MadNLP.input_type(::Type{CUDSSSolver}) = :csc
MadNLP.default_options(::Type{CUDSSSolver}) = CudssSolverOptions()
MadNLP.is_inertia(M::CUDSSSolver) = (M.opt.cudss_algorithm ∈ (MadNLP.CHOLESKY, MadNLP.LDL))
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
