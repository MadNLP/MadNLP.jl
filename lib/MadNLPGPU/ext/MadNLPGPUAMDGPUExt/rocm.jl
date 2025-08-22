#=
    MadNLP.MadNLPOptions
=#

function MadNLP.MadNLPOptions{T}(
    nlp::MadNLP.AbstractNLPModel{T,VT};
    dense_callback = MadNLP.is_dense_callback(nlp),
    callback = dense_callback ? MadNLP.DenseCallback : MadNLP.SparseCallback,
    kkt_system = dense_callback ? MadNLP.DenseCondensedKKTSystem : MadNLP.SparseCondensedKKTSystem,
    linear_solver = MadNLPGPU.LapackROCSolver,
    tol = MadNLP.get_tolerance(T,kkt_system),
    bound_relax_factor = tol,
) where {T, VT <: ROCVector{T}}
    return MadNLP.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
    )
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
    ROCSparseMatrixCSC to ROCMatrix
=#

function MadNLPGPU.gpu_transfer!(y::ROCMatrix{T}, x::rocSPARSE.ROCSparseMatrixCSC{T}) where {T}
    n = size(y, 2)
    fill!(y, zero(T))
    backend = ROCBackend()
    MadNLPGPU._csc_to_dense_kernel!(backend)(y, x.colPtr, x.rowVal, x.nzVal, ndrange = n)
    synchronize(backend)
    return
end
