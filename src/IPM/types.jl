mutable struct RobustRestorer{T, VT}
    obj_val_R::T
    f_R::VT
    x_ref::VT

    theta_ref::T
    D_R::VT
    obj_val_R_trial::T

    pp::VT
    nn::VT
    zp::VT
    zn::VT

    dpp::VT
    dnn::VT
    dzp::VT
    dzn::VT

    pp_trial::VT
    nn_trial::VT

    inf_pr_R::T
    inf_du_R::T
    inf_compl_R::T

    mu_R::T
    tau_R::T
    zeta::T

    filter::Vector{Tuple{T,T}}
end

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
