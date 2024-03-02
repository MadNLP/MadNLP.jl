import CUDSS
import SparseArrays

@kwdef mutable struct CudssSolverOptions <: MadNLP.AbstractOptions
    cudss_algorithm::MadNLP.LinearFactorization = MadNLP.LDL
end

mutable struct CUDSSSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing, CUDSS.CudssSolver}
    tril::CUSPARSE.CuSparseMatrixCSC{T}
    x_gpu::CUDA.CuVector{T}
    b_gpu::CUDA.CuVector{T}

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
    structure = if opt.cudss_algorithm == MadNLP.LU
        "G"
    elseif opt.cudss_algorithm == MadNLP.CHOLESKY
        "SPD"
    elseif opt.cudss_algorithm == MadNLP.LDL
        "S"
    end

    matrix = CUDSS.CudssMatrix(
        CUSPARSE.CuSparseMatrixCSR(csc.colPtr, csc.rowVal, csc.nzVal, csc.dims), 
        structure,
        view
    )
    
    # TODO: pass config options here.
    config = CUDSS.CudssConfig()
    data = CUDSS.CudssData()
    
    solver = CUDSS.CudssSolver(matrix, config, data)

    x_gpu = CUDA.zeros(T, n)
    b_gpu = CUDA.zeros(T, n)
    
    CUDSS.cudss("analysis", solver, x_gpu, b_gpu)

    return CUDSSSolver(
        solver, csc,
        # full, tril_to_full_view,
        x_gpu, b_gpu,
        opt, logger
    )
end

function MadNLP.factorize!(M::CUDSSSolver)
    # copyto!(M.full.nzVal, M.tril_to_full_view)
    CUDSS.cudss_set(M.inner.matrix, SparseArrays.nonzeros(M.tril))
    CUDSS.cudss("factorization", M.inner, M.x_gpu, M.b_gpu)

    return M
end

function MadNLP.solve!(M::CUDSSSolver{T}, x) where T
    CUDSS.cudss("solve", M.inner, M.x_gpu, x)
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
            return (0, n, 0)
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
MadNLP.introduce(M::CUDSSSolver) = "cuDSS"

