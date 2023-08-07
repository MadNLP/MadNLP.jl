abstract type AbstractInertiaCorrector end
struct InertiaAuto <: AbstractInertiaCorrector end
struct InertiaBased <: AbstractInertiaCorrector end
struct InertiaIgnore <: AbstractInertiaCorrector end
struct InertiaFree{
    T,
    VT <: AbstractVector{T},
    KKTVec <: AbstractKKTVector{T, VT}
} <: AbstractInertiaCorrector 
    p0::KKTVec
    d0::KKTVec
    t::VT
    wx::VT
    g::VT
end

function build_inertia_corrector(::Type{InertiaBased}, ::Type{VT}, n, m, nlb, nub, ind_lb, ind_ub) where VT
    return InertiaBased()
end
function build_inertia_corrector(::Type{InertiaIgnore}, ::Type{VT}, n, m, nlb, nub, ind_lb, ind_ub) where VT
    return InertiaIgnore()
end
function build_inertia_corrector(::Type{InertiaFree}, ::Type{VT}, n, m, nlb, nub, ind_lb, ind_ub) where VT
    p0 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    d0 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    t = VT(undef, n)
    wx= VT(undef, n)
    g = VT(undef, n)
    
    return InertiaFree(
        p0, d0, t, wx, g
    )
end
