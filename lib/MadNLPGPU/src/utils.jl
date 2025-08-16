@enum ORDERING begin
    DEFAULT_ORDERING = 0
    METIS_ORDERING = 1
    AMD_ORDERING = 2
    USER_ORDERING = 3
    SYMAMD_ORDERING = 4
    COLAMD_ORDERING = 5
end

function MadNLP._madnlp_unsafe_wrap(vec::VT, n, shift=1) where {T, VT <: CuVector{T}}
    return view(vec,shift:shift+n-1)
end

# Local transfer! function to move data on the device.
transfer!(x::AbstractArray, y::AbstractArray) = copyto!(x, y)

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
    SparseMatrixCOO to CuSparseMatrixCSC
=#

function transfer!(
    dest::CUSPARSE.CuSparseMatrixCSC,
    src::MadNLP.SparseMatrixCOO,
    map,
)
    return copyto!(view(dest.nzVal, map), src.V)
end

#=
    CuSparseMatrixCSC to CuMatrix
=#

function transfer!(y::CuMatrix{T}, x::CUSPARSE.CuSparseMatrixCSC{T}) where {T}
    n = size(y, 2)
    fill!(y, zero(T))
    backend = CUDABackend()
    _csc_to_dense_kernel!(backend)(y, x.colPtr, x.rowVal, x.nzVal, ndrange = n)
    synchronize(backend)
    return
end

# CUBLAS operations
MadNLP.symul!(y, A, x::CuVector{T}, α = one(T), β = zero(T)) where T = CUBLAS.symv!('L', T(α), A, x, T(β), y)
MadNLP._ger!(alpha::Number, x::CuVector{T}, y::CuVector{T}, A::CuMatrix{T}) where T = CUBLAS.ger!(alpha, x, y, A)
