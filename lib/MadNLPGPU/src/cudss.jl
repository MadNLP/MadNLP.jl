import CUDSS
import SparseArrays

@kwdef mutable struct CudssSolverOptions <: MadNLP.AbstractOptions
    cudss_algorithm::MadNLP.LinearFactorization = MadNLP.LU
end

mutable struct CUDSSSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing, CUDSS.CudssSolver}
    tril::CUSPARSE.CuSparseMatrixCSC{T}
    full::CUSPARSE.CuSparseMatrixCSR{T}
    tril_to_full_view::CuSubVector{T}
    x_gpu::CUDA.CuVector{T}
    b_gpu::CUDA.CuVector{T}

    # Permutation
    full_permuted::CUSPARSE.CuSparseMatrixCSR{T}
    perm::CUDA.CuVector{Int}
    perm_nzval::CUDA.CuVector{Int}

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

    full,tril_to_full_view = MadNLP.get_tril_to_full(csc)

    full = CUSPARSE.CuSparseMatrixCSR(
        full.colPtr,
        full.rowVal,
        full.nzVal,
        full.dims
    )

    # Reordering
    full_cpu = MadNLP.SparseMatrixCSC(full)
    full_cpu_order = MadNLP.SparseMatrixCSC(
        n, m,
        full_cpu.colptr,
        full_cpu.rowval,
        Array(1:length(full_cpu.nzval)),
    )
    perm = AMD.amd(full_cpu_order)
    full_cpu_reorder = full_cpu_order[perm, perm]
    pnzval = full_cpu_reorder.nzval

    full_permuted = CUSPARSE.CuSparseMatrixCSR(
        CuArray(full_cpu_reorder.colptr),
        CuArray(full_cpu_reorder.rowval),
        similar(full.nzVal),
        (n, m),
    )

    view = 'L'
    structure = if opt.cudss_algorithm == MadNLP.LU
        "G"
    elseif opt.cudss_algorithm == MadNLP.CHOLESKY
        "SPD"
    elseif opt.cudss_algorithm == MadNLP.BUNCHKAUFMAN
        "S"
    end

    matrix = CUDSS.CudssMatrix(full_permuted, structure, view)
    # TODO: pass config options here.
    config = CUDSS.CudssConfig()
    data = CUDSS.CudssData()
    solver = CUDSS.CudssSolver(matrix, config, data)

    x_gpu = CUDA.zeros(T, n)
    b_gpu = CUDA.zeros(T, n)

    CUDSS.cudss("analysis", solver, x_gpu, b_gpu)

    g_perm = CuVector{Int}(perm)
    g_pnzval = CuVector{Int}(pnzval)

    return CUDSSSolver(
        solver, csc, full, tril_to_full_view,
        x_gpu, b_gpu,
        full_permuted, g_perm, g_pnzval,
        opt, logger
    )
end

function MadNLP.factorize!(M::CUDSSSolver)
    copyto!(M.full.nzVal, M.tril_to_full_view)
    # Permutation
    _copy_to_perm_2!(CUDABackend())(M.full_permuted.nzVal, M.perm_nzval, M.full.nzVal; ndrange= length(M.perm_nzval))
    synchronize(CUDABackend())

    CUDSS.cudss_set(M.inner.matrix, SparseArrays.nonzeros(M.full_permuted))
    CUDSS.cudss("factorization", M.inner, M.x_gpu, M.b_gpu)
    return M
end

function MadNLP.solve!(M::CUDSSSolver{T}, x) where T
    copyto!(M.b_gpu, x)
    _copy_to_perm_2!(CUDABackend())(M.b_gpu, M.perm, x; ndrange=length(M.perm))
    synchronize(CUDABackend())
    CUDSS.cudss("solve", M.inner, M.x_gpu, M.b_gpu)
    _copy_to_perm!(CUDABackend())(x, M.perm, M.x_gpu; ndrange=length(M.perm))
    synchronize(CUDABackend())
    return x
end

MadNLP.input_type(::Type{CUDSSSolver}) = :csc
MadNLP.default_options(::Type{CUDSSSolver}) = CudssSolverOptions()
MadNLP.is_inertia(M::CUDSSSolver) = (M.opt.cudss_algorithm ∈ (MadNLP.CHOLESKY, MadNLP.BUNCHKAUFMAN))
function inertia(M::CUDSSSolver)
    n = size(M.full, 1)
    if M.opt.cudss_algorithm == MadNLP.CHOLESKY
        info = CUDSS.cudss_get(M.inner, "info")
        if info == 0
            return (n, 0, 0)
        else
            return (0, n, 0)
        end
    elseif M.opt.cudss_algorithm == MadNLP.BUNCHKAUFMAN
        # N.B.: cuDSS does not always return the correct inertia.
        (k, l) = CUDSS.cudss_get(M.inner, "inertia")
        return (k, n - k - l, l)
    end
end

MadNLP.improve!(M::CUDSSSolver) = false
MadNLP.is_supported(::Type{CUDSSSolver},::Type{Float32}) = true
MadNLP.is_supported(::Type{CUDSSSolver},::Type{Float64}) = true
MadNLP.introduce(M::CUDSSSolver) = "cuDSS"

