@enum ORDERING begin
    DEFAULT_ORDERING = 0
    METIS_ORDERING = 1
    AMD_ORDERING = 2
    USER_ORDERING = 3
end

function MadNLP._madnlp_unsafe_wrap(vec::VT, n, shift=1) where {T, VT <: CuVector{T}}
    return view(vec,shift:shift+n-1)
end

# Local transfer! function to move data on the device.
transfer!(x::AbstractArray, y::AbstractArray) = copyto!(x, y)

#=
    copyto!
=#

@kernel function _copy_to_map_kernel!(y, p, x)
    i = @index(Global)
    @inbounds y[p[i]] = x[i]
end

@kernel function _copy_from_map_kernel!(y, x, p)
    i = @index(Global)
    @inbounds y[i] = x[p[i]]
end

#=
    SparseMatrixCSC to CuSparseMatrixCSC
=#

function CUSPARSE.CuSparseMatrixCSC{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    return CUSPARSE.CuSparseMatrixCSC{Tv,Ti}(
        CuArray(A.colptr),
        CuArray(A.rowval),
        CuArray(A.nzval),
        size(A),
    )
end

#=
    SparseMatrixCOO to CuSparseMatrixCSC
=#

function MadNLP.transfer!(
    dest::CUSPARSE.CuSparseMatrixCSC,
    src::MadNLP.SparseMatrixCOO,
    map,
)
    return copyto!(view(dest.nzVal, map), src.V)
end

#=
    CuSparseMatrixCSC to CuMatrix
=#

@kernel function _csc_to_dense_kernel!(y, @Const(colptr), @Const(rowval), @Const(nzval))
    col = @index(Global)
    @inbounds for ptr in colptr[col]:colptr[col+1]-1
        row = rowval[ptr]
        y[row, col] = nzval[ptr]
    end
end

function transfer!(y::CuMatrix{T}, x::CUSPARSE.CuSparseMatrixCSC{T}) where {T}
    n = size(y, 2)
    fill!(y, zero(T))
    _csc_to_dense_kernel!(CUDABackend())(y, x.colPtr, x.rowVal, x.nzVal, ndrange = n)
    synchronize(CUDABackend())
    return
end

# BLAS operations
symul!(y, A, x::CuVector{T}, α = 1., β = 0.) where T = CUBLAS.symv!('L', T(α), A, x, T(β), y)

MadNLP._ger!(alpha::Number, x::CuVector{T}, y::CuVector{T}, A::CuMatrix{T}) where T = CUBLAS.ger!(alpha, x, y, A)

