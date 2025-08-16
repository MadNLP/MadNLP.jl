function MadNLP.MadNLPOptions{T}(
    nlp::MadNLP.AbstractNLPModel{T,VT};
    callback = MadNLP.DenseCallback,
    kkt_system = MadNLP.DenseCondensedKKTSystem,
    linear_solver = LapackROCSolver,
    tol = MadNLP.get_tolerance(T,kkt_system),
    bound_relax_factor = tol
) where {T, VT <: ROCVector{T}}
    return MadNLP.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
    )
end

function MadNLP._madnlp_unsafe_wrap(vec::VT, n, shift=1) where {T, VT <: ROCVector{T}}
    return view(vec,shift:shift+n-1)
end

#=
    SparseMatrixCSC to ROCSparseMatrixCSC
=#

function rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    return rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}(
        ROCVector(A.colptr),
        ROCVector(A.rowval),
        ROCVector(A.nzval),
        size(A),
    )
end

#=
    SparseMatrixCOO to ROCSparseMatrixCSC
=#

function MadNLP.transfer!(
    dest::rocSPARSE.ROCSparseMatrixCSC,
    src::MadNLP.SparseMatrixCOO,
    map,
)
    return copyto!(view(dest.nzVal, map), src.V)
end

#=
    ROCSparseMatrixCSC to ROCMatrix
=#

function MadNLP.transfer!(y::ROCMatrix{T}, x::rocSPARSE.ROCSparseMatrixCSC{T}) where {T}
    n = size(y, 2)
    fill!(y, zero(T))
    backend = ROCBackend()
    _csc_to_dense_kernel!(backend)(y, x.colPtr, x.rowVal, x.nzVal, ndrange = n)
    synchronize(backend)
    return
end

MadNLP.symul!(y, A, x::ROCVector{T}, α = one(T), β = zero(T)) where T = rocBLAS.symv!('L', T(α), A, x, T(β), y)
MadNLP._ger!(alpha::Number, x::ROCVector{T}, y::ROCVector{T}, A::ROCMatrix{T}) where T = rocBLAS.ger!(alpha, x, y, A)
