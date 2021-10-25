

abstract type AbstractKKTVector{T, VT} end

#=
    ReducedKKTVector
=#

struct ReducedKKTVector{T, VT<:AbstractVector{T}} <: AbstractKKTVector{T, VT}
    x::VT
    xp::VT # unsafe view
    xl::VT # unsafe view
end

ReducedKKTVector(n::Int, m::Int, nlb::Int, nub::Int) = ReducedKKTVector(n, m)
function ReducedKKTVector(n::Int, m::Int)
    x = Vector{Float64}(undef, n + m)
    # Wrap directly array x to avoid dealing with views
    pp = pointer(x)
    xp = unsafe_wrap(Vector{Float64}, pp, n)
    pd = pointer(x, n + 1)
    xl = unsafe_wrap(Vector{Float64}, pd, m)
    return ReducedKKTVector{Float64, Vector{Float64}}(x, xp, xl)
end

Base.length(rhs::ReducedKKTVector) = length(rhs.x)


#=
    UnreducedKKTVector
=#

struct UnreducedKKTVector{T, VT<:AbstractVector{T}} <: AbstractKKTVector{T, VT}
    values::VT
    x::VT  # unsafe view
    xp::VT # unsafe view
    xl::VT # unsafe view
    xzl::VT # unsafe view
    xzu::VT # unsafe view
end

function UnreducedKKTVector(n::Int, m::Int, nlb::Int, nub::Int)
    values = Vector{Float64}(undef, n + m + nlb + nub)
    # Wrap directly array x to avoid dealing with views
    pp = pointer(values)
    x = unsafe_wrap(Vector{Float64}, pp, n + m) # Primal-Dual
    xp = unsafe_wrap(Vector{Float64}, pp, n) # Primal
    pd = pointer(values, n + 1)
    xl = unsafe_wrap(Vector{Float64}, pd, m) # Dual
    pzl = pointer(values, n + m + 1)
    xzl = unsafe_wrap(Vector{Float64}, pzl, nlb) # Lower bound
    pzu = pointer(values, n + m + nlb + 1)
    xzu = unsafe_wrap(Vector{Float64}, pzu, nub) # Upper bound
    return UnreducedKKTVector{Float64, Vector{Float64}}(values, x, xp, xl, xzl, xzu)
end

Base.length(rhs::UnreducedKKTVector) = length(rhs.values)

