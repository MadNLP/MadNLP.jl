# Backend-agnostic data-transfer kernels (generalized from HybridKKT.jl's
# CUDA-specific versions: CuVector -> AbstractGPUVector, CUDABackend() ->
# get_backend(x)), so they run on any KernelAbstractions backend (CUDA, AMDGPU,
# ...) without a hard CUDA dependency. CPU fallbacks dispatch on AbstractVector.

# ---- CPU implementations ----

"Implement `dest .= src[idx]`."
function index_copy!(dest::AbstractVector{T}, src::AbstractVector{T}, idx::AbstractVector{Ti}) where {T, Ti <: Integer}
    @assert length(dest) == length(idx)
    return @inbounds for i in eachindex(idx)
        dest[i] = src[idx[i]]
    end
end

"Implement `dest[idx] .= src`."
function index_copy!(dest::AbstractVector{T}, idx::AbstractVector{Ti}, src::AbstractVector{T}) where {T, Ti <: Integer}
    @assert length(src) == length(idx)
    return @inbounds for i in eachindex(idx)
        dest[idx[i]] = src[i]
    end
end

"Implement `dest[idx] .= val`."
function fixed!(dest::AbstractVector{T}, idx::AbstractVector{Ti}, val::T) where {T, Ti <: Integer}
    return @inbounds for i in idx
        dest[i] = val
    end
end

function transfer_coef!(G::SparseMatrixCSC, map::Vector{Int}, coefs::Vector{Tv}, ind_eq) where {Tv}
    valsG = nonzeros(G)
    fill!(valsG, zero(Tv))
    for k in 1:length(map)
        valsG[map[k]] += coefs[ind_eq[k]]
    end
    return
end

# ---- KernelAbstractions implementations (any GPU backend) ----

@kernel function _copy_index_from_kernel!(dest, src, idx)
    i = @index(Global, Linear)
    @inbounds dest[i] = src[idx[i]]
end
function index_copy!(dest::AbstractGPUVector{T}, src::AbstractGPUVector{T}, idx::AbstractGPUVector{Ti}) where {T, Ti <: Integer}
    @assert length(dest) == length(idx)
    return if length(dest) > 0
        backend = get_backend(dest)
        _copy_index_from_kernel!(backend)(dest, src, idx; ndrange = length(dest))
        KernelAbstractions.synchronize(backend)
    end
end

@kernel function _copy_index_to_kernel!(dest, src, idx)
    i = @index(Global, Linear)
    @inbounds dest[idx[i]] = src[i]
end
function index_copy!(dest::AbstractGPUVector{T}, idx::AbstractGPUVector{Ti}, src::AbstractGPUVector{T}) where {T, Ti <: Integer}
    @assert length(src) == length(idx)
    return if length(src) > 0
        backend = get_backend(dest)
        _copy_index_to_kernel!(backend)(dest, src, idx; ndrange = length(src))
        KernelAbstractions.synchronize(backend)
    end
end

@kernel function _fixed_kernel!(dest, idx, val)
    i = @index(Global, Linear)
    dest[idx[i]] = val
end
function fixed!(dest::AbstractGPUVector{T}, idx::AbstractGPUVector{Ti}, val::T) where {T, Ti <: Integer}
    length(idx) == 0 && return
    backend = get_backend(dest)
    _fixed_kernel!(backend)(dest, idx, val; ndrange = length(idx))
    return KernelAbstractions.synchronize(backend)
end

@kernel function _transfer_coef_kernel!(valsG, to_map, valJ, fr_map)
    k = @index(Global, Linear)
    @inbounds begin
        Atomix.@atomic valsG[to_map[k]] += valJ[fr_map[k]]
    end
end
# `G` is a GPU sparse matrix (its nonzeros are an AbstractGPUVector); dispatch on
# `map` so we stay independent of the concrete sparse type.
function transfer_coef!(G, map::AbstractGPUVector{Int}, coefs::AbstractGPUVector{Tv}, ind_eq) where {Tv}
    valsG = nonzeros(G)
    fill!(valsG, zero(Tv))
    backend = get_backend(map)
    _transfer_coef_kernel!(backend)(valsG, map, coefs, ind_eq; ndrange = length(map))
    return
end
