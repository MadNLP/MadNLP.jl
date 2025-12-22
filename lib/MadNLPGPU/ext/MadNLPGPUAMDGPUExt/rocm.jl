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

if VERSION > v"1.11" # See https://github.com/JuliaGPU/AMDGPU.jl/issues/607. 1-norm of view() of RocArray seems to have been broken in v1.12
    function MadNLP.get_sd(l::ROCVector{T}, zl_r, zu_r, s_max) where T
        return max(
            s_max,
            (my1norm(l)+my1norm(zl_r)+my1norm(zu_r)) / max(1, (length(l)+length(zl_r)+length(zu_r))),
        ) / s_max
    end
    function MadNLP.get_sc(zl_r::SubArray{T,1,VT}, zu_r, s_max) where {T, VT <: ROCVector{T}}
        return max(
            s_max,
            (my1norm(zl_r)+my1norm(zu_r)) / max(1,length(zl_r)+length(zu_r)),
        ) / s_max
    end
    my1norm(x) = mapreduce(abs, +, x)
end

#=
    MadNLP._ger!
=#

MadNLP._ger!(alpha::T, x::ROCVector{T}, y::ROCVector{T}, A::ROCMatrix{T}) where T = rocBLAS.ger!(alpha, x, y, A)

#=
    MadNLP._syr!
=#

MadNLP._syr!(uplo::Char, alpha::T, x::ROCVector{T}, A::ROCMatrix{T}) where T = rocBLAS.syr!(uplo, alpha, x, A)

#=
    MadNLP._symv!
=#

MadNLP._symv!(uplo::Char, alpha::T, A::ROCMatrix{T}, x::ROCVector{T}, beta::T, y::ROCVector{T}) where T = rocBLAS.symv!(uplo, alpha, A, x, beta, y)

#=
    MadNLP._syrk!
=#

MadNLP._syrk!(uplo::Char, trans::Char, alpha::T, A::ROCMatrix{T}, beta::T, C::ROCMatrix{T}) where T = rocBLAS.syrk!(uplo, trans, alpha, A, beta, C)
