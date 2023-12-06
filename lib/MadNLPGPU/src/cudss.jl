import CUDSS
import SparseArrays

@kwdef mutable struct CudssSolverOptions <: MadNLP.AbstractOptions
    cudss_algorithm::MadNLP.LinearFactorization = MadNLP.LU
end

mutable struct CUDSSSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing, CUDSS.CudssSolver}
    tril::SparseArrays.SparseMatrixCSC{T}
    full::SparseArrays.SparseMatrixCSC{T}
    tril_to_full_view::MadNLP.SubVector{T}
    K::CUSPARSE.CuSparseMatrixCSR{T}
    x_gpu::CUDA.CuVector{T}
    b_gpu::CUDA.CuVector{T}

    opt::CudssSolverOptions
    logger::MadNLP.MadNLPLogger
end

function CUDSSSolver(
    csc::SparseArrays.SparseMatrixCSC{T};
    opt=CudssSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
) where T
    n, m = size(csc)
    @assert n == m

    full,tril_to_full_view = MadNLP.get_tril_to_full(csc)
    full.nzval .= tril_to_full_view

    K_gpu = CUSPARSE.CuSparseMatrixCSR(full)

    view = 'L'
    structure = if opt.cudss_algorithm == MadNLP.LU
        "G"
    elseif opt.cudss_algorithm == MadNLP.CHOLESKY
        "SPD"
    elseif opt.cudss_algorithm == MadNLP.BUNCHKAUFMAN
        "S"
    end

    matrix = CUDSS.CudssMatrix(K_gpu, structure, view)
    # TODO: pass config options here.
    config = CUDSS.CudssConfig()
    data = CUDSS.CudssData()
    solver = CUDSS.CudssSolver(matrix, config, data)

    x_gpu = CUDA.zeros(T, n)
    b_gpu = CUDA.zeros(T, n)

    CUDSS.cudss("analysis", solver, x_gpu, b_gpu)

    return CUDSSSolver(
        solver, csc, full, tril_to_full_view, K_gpu,
        x_gpu, b_gpu,
        opt, logger
    )
end

function MadNLP.factorize!(M::CUDSSSolver)
    nzvals = SparseArrays.nonzeros(M.full)
    copyto!(nzvals, M.tril_to_full_view)
    copyto!(SparseArrays.nonzeros(M.K), nzvals)
    CUDSS.cudss_set(M.inner.matrix, SparseArrays.nonzeros(M.K))
    CUDSS.cudss("factorization", M.inner, M.x_gpu, M.b_gpu)
    return M
end

function MadNLP.solve!(M::CUDSSSolver{T}, x) where T
    copyto!(M.b_gpu, x)
    CUDSS.cudss("solve", M.inner, M.x_gpu, M.b_gpu)
    copyto!(x, M.x_gpu)
    return x
end

MadNLP.input_type(::Type{CUDSSSolver}) = :csc
MadNLP.default_options(::Type{CUDSSSolver}) = CudssSolverOptions()
MadNLP.is_inertia(M::CUDSSSolver) = false
MadNLP.improve!(M::CUDSSSolver) = false
MadNLP.is_supported(::Type{CUDSSSolver},::Type{Float32}) = true
MadNLP.is_supported(::Type{CUDSSSolver},::Type{Float64}) = true
MadNLP.introduce(M::CUDSSSolver) = "cuDSS"

