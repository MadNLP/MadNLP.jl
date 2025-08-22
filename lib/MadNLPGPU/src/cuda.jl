#=
    MadNLP.MadNLPOptions
=#

function MadNLP.MadNLPOptions{T}(
    nlp::AbstractNLPModel{T,VT};
    dense_callback = MadNLP.is_dense_callback(nlp),
    callback = dense_callback ? MadNLP.DenseCallback : MadNLP.SparseCallback,
    kkt_system = dense_callback ? MadNLP.DenseCondensedKKTSystem : MadNLP.SparseCondensedKKTSystem,
    linear_solver = dense_callback ? LapackGPUSolver : CUDSSSolver,
    tol = MadNLP.get_tolerance(T,kkt_system),
    bound_relax_factor = (kkt_system == MadNLP.SparseCondensedKKTSystem) ? tol : T(1e-8),
) where {T, VT <: CuVector{T}}
    return MadNLP.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
    )
end

#=
    SparseMatrixCSC to CuSparseMatrixCSC
=#

function CUSPARSE.CuSparseMatrixCSC{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    return CUSPARSE.CuSparseMatrixCSC{Tv,Ti}(
        CuVector(A.colptr),
        CuVector(A.rowval),
        CuVector(A.nzval),
        size(A),
    )
end

#=
    CuSparseMatrixCSC to CuMatrix
=#

function gpu_transfer!(y::CuMatrix{T}, x::CUSPARSE.CuSparseMatrixCSC{T}) where {T}
    n = size(y, 2)
    fill!(y, zero(T))
    backend = CUDABackend()
    _csc_to_dense_kernel!(backend)(y, x.colPtr, x.rowVal, x.nzVal, ndrange = n)
    synchronize(backend)
    return
end
