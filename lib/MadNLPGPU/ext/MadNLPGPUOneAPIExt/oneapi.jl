#=
    MadNLP.MadNLPOptions
=#

function MadNLP.MadNLPOptions{T}(
    nlp::MadNLP.AbstractNLPModel{T,VT};
    dense_callback = MadNLP.is_dense_callback(nlp),
    callback = dense_callback ? MadNLP.DenseCallback : MadNLP.SparseCallback,
    kkt_system = dense_callback ? MadNLP.DenseCondensedKKTSystem : MadNLP.SparseCondensedKKTSystem,
    linear_solver = MadNLPGPU.LapackOneMKLSolver,
    tol = MadNLP.get_tolerance(T,kkt_system),
    bound_relax_factor = tol,
) where {T, VT <: oneVector{T}}
    return MadNLP.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
    )
end

#=
    SparseMatrixCSC to oneSparseMatrixCSC
=#

function oneMKL.oneSparseMatrixCSC{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    return oneMKL.oneSparseMatrixCSC{Tv,Ti}(
        oneVector(A.colptr),
        oneVector(A.rowval),
        oneVector(A.nzval),
        size(A),
    )
end

#=
    oneSparseMatrixCSC to oneMatrix
=#

function MadNLPGPU.gpu_transfer!(y::oneMatrix{T}, x::oneMKL.oneSparseMatrixCSC{T}) where {T}
    n = size(y, 2)
    fill!(y, zero(T))
    backend = oneAPIBackend()
    MadNLPGPU._csc_to_dense_kernel!(backend)(y, x.colPtr, x.rowVal, x.nzVal, ndrange = n)
    synchronize(backend)
    return
end
