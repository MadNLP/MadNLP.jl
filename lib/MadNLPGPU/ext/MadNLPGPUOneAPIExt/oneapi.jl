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

#=
    MadNLP._syr!
=#

MadNLP._syr!(uplo::Char, alpha::T, x::oneVector{T}, A::oneMatrix{T}) where T = oneMKL.syr!(uplo, alpha, x, A)

#=
    MadNLP._symv!
=#

MadNLP._symv!(uplo::Char, alpha::T, A::oneMatrix{T}, x::oneVector{T}, beta::T, y::oneVector{T}) where T = oneMKL.symv!(uplo, alpha, A, x, beta, y)

#=
    MadNLP._syrk!
=#

MadNLP._syrk!(uplo::Char, trans::Char, alpha::T, A::oneMatrix{T}, beta::T, C::oneMatrix{T}) where T = oneMKL.syrk!(uplo, trans, alpha, A, beta, C)

#=
    MadNLP._trsm!
=#

MadNLP._trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::T, A::oneMatrix{T}, B::oneMatrix{T}) where T = oneMKL.trsm!(side, uplo, transa, diag, alpha, A, B)

#=
    MadNLP._dgmm!
=#

MadNLP._dgmm!(side::Char, A::oneMatrix{T}, x::oneVector{T}, B::oneMatrix{T}) where T = oneMKL.dgmm!(side, A, x, B)
