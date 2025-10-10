
# KKT system updates -------------------------------------------------------
# Set diagonal
function set_aug_diagonal!(kkt::AbstractKKTSystem{T}, solver::AbstractMadNLPSolver{T}) where T
    x = full(_x(solver))
    xl = full(_xl(solver))
    xu = full(_xu(solver))
    zl = full(_zl(solver))
    zu = full(_zu(solver))

    fill!(kkt.reg, zero(T))
    fill!(kkt.du_diag, zero(T))
    kkt.l_diag .= _xl_r(solver) .- _x_lr(solver)   # (Xˡ - X)
    kkt.u_diag .= _x_ur(solver) .- _xu_r(solver)   # (X - Xᵘ)
    copyto!(kkt.l_lower, _zl_r(solver))
    copyto!(kkt.u_lower, _zu_r(solver))

    _set_aug_diagonal!(kkt)
    return
end

function _set_aug_diagonal!(kkt::AbstractKKTSystem)
    copyto!(kkt.pr_diag, kkt.reg)
    kkt.pr_diag[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    kkt.pr_diag[kkt.ind_ub] .-= kkt.u_lower ./ kkt.u_diag
    return
end

function _set_aug_diagonal!(kkt::AbstractUnreducedKKTSystem)
    copyto!(kkt.pr_diag, kkt.reg)
    kkt.l_lower_aug .= sqrt.(kkt.l_lower)
    kkt.u_lower_aug .= sqrt.(kkt.u_lower)
    return
end

function set_aug_diagonal!(kkt::ScaledSparseKKTSystem{T}, solver::AbstractMadNLPSolver{T}) where T
    fill!(kkt.reg, zero(T))
    fill!(kkt.du_diag, zero(T))
    # Ensure l_diag and u_diag have only non negative entries
    kkt.l_diag .= _x_lr(solver) .- _xl_r(solver)   # (X - Xˡ)
    kkt.u_diag .= _xu_r(solver) .- _x_ur(solver)   # (Xᵘ - X)
    copyto!(kkt.l_lower, _zl_r(solver))
    copyto!(kkt.u_lower, _zu_r(solver))
    _set_aug_diagonal!(kkt)
end

function _set_aug_diagonal!(kkt::ScaledSparseKKTSystem{T}) where T
    xlzu = kkt.buffer1
    xuzl = kkt.buffer2
    fill!(xlzu, zero(T))
    fill!(xuzl, zero(T))

    xlzu[kkt.ind_ub] .= kkt.u_lower    # zᵘ
    xlzu[kkt.ind_lb] .*= kkt.l_diag    # (X - Xˡ) zᵘ

    xuzl[kkt.ind_lb] .= kkt.l_lower    # zˡ
    xuzl[kkt.ind_ub] .*= kkt.u_diag    # (Xᵘ - X) zˡ

    kkt.pr_diag .= xlzu .+ xuzl

    fill!(kkt.scaling_factor, one(T))
    kkt.scaling_factor[kkt.ind_lb] .*= sqrt.(kkt.l_diag)
    kkt.scaling_factor[kkt.ind_ub] .*= sqrt.(kkt.u_diag)

    # Scale regularization by scaling factor.
    kkt.pr_diag .+= kkt.reg .* kkt.scaling_factor.^2
    return
end


# Robust restoration
function set_aug_RR!(kkt::AbstractKKTSystem, solver::AbstractMadNLPSolver, RR::RobustRestorer)
    x = full(_x(solver))
    xl = full(_xl(solver))
    xu = full(_xu(solver))
    zl = full(_zl(solver))
    zu = full(_zu(solver))
    kkt.reg .= RR.zeta .* RR.D_R .^ 2
    kkt.du_diag .= .- RR.pp ./ RR.zp .- RR.nn ./ RR.zn
    copyto!(kkt.l_lower, _zl_r(solver))
    copyto!(kkt.u_lower, _zu_r(solver))
    kkt.l_diag .= _xl_r(solver) .- _x_lr(solver)
    kkt.u_diag .= _x_ur(solver) .- _xu_r(solver)

    _set_aug_diagonal!(kkt)
    return
end

function set_aug_RR!(kkt::ScaledSparseKKTSystem, solver::AbstractMadNLPSolver, RR::RobustRestorer)
    x = full(_x(solver))
    xl = full(_xl(solver))
    xu = full(_xu(solver))
    zl = full(_zl(solver))
    zu = full(_zu(solver))
    kkt.reg .= RR.zeta .* RR.D_R .^ 2
    kkt.du_diag .= .- RR.pp ./ RR.zp .- RR.nn ./ RR.zn
    copyto!(kkt.l_lower, _zl_r(solver))
    copyto!(kkt.u_lower, _zu_r(solver))
    kkt.l_diag .= _x_lr(solver) .- _xl_r(solver)
    kkt.u_diag .= _xu_r(solver) .- _x_ur(solver)

    _set_aug_diagonal!(kkt)
    return
end

function set_f_RR!(solver::AbstractMadNLPSolver, RR::RobustRestorer)
    x = full(_x(solver))
    RR.f_R .= RR.zeta .* RR.D_R .^ 2 .* (x .- RR.x_ref)
    return
end

# Set RHS
function set_aug_rhs!(solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, c::AbstractVector)
    px = primal(_p(solver))
    x = primal(_x(solver))
    f = primal(_f(solver))
    xl = primal(_xl(solver))
    xu = primal(_xu(solver))
    zl = full(_zl(solver))
    zu = full(_zu(solver))
    py = dual(_p(solver))
    pzl = dual_lb(_p(solver))
    pzu = dual_ub(_p(solver))

    px .= .-f .+ zl .- zu .- _jacl(solver)
    py .= .-c
    pzl .= (_xl_r(solver) .- _x_lr(solver)) .* _zl_r(solver) .+ _mu(solver)
    pzu .= (_xu_r(solver) .- _x_ur(solver)) .* _zu_r(solver) .- _mu(solver)
    return
end

# Set RHS RR
function set_aug_rhs_RR!(
    solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, RR::RobustRestorer, rho,
)
    x = full(_x(solver))
    xl = full(_xl(solver))
    xu = full(_xu(solver))
    zl = full(_zl(solver))
    zu = full(_zu(solver))

    px = primal(_p(solver))
    py = dual(_p(solver))
    pzl = dual_lb(_p(solver))
    pzu = dual_ub(_p(solver))

    mu = RR.mu_R

    px .= .- RR.f_R .+ zl .- zu .- _jacl(solver)
    py .= .- _c(solver) .+ RR.pp .- RR.nn .+
        (mu .- (rho .- _y(solver)) .* RR.pp) ./ RR.zp .-
        (mu .- (rho .+ _y(solver)) .* RR.nn) ./ RR.zn

    pzl .= (_xl_r(solver) .- _x_lr(solver)) .* _zl_r(solver) .+ mu
    pzu .= (_xu_r(solver) .- _x_ur(solver)) .* _zu_r(solver) .- mu

    return
end

# solving KKT system
@inbounds function _kktmul!(
    w::AbstractKKTVector,
    x::AbstractKKTVector,
    reg,
    du_diag,
    l_lower,
    u_lower,
    l_diag,
    u_diag,
    alpha,
    beta,
)
    primal(w) .+= alpha .* reg .* primal(x)
    dual(w) .+= alpha .* du_diag .* dual(x)
    w.xp_lr .-= alpha .* dual_lb(x)
    w.xp_ur .+= alpha .* dual_ub(x)
    dual_lb(w) .= beta .* dual_lb(w) .+ alpha .* (x.xp_lr .* l_lower .- dual_lb(x) .* l_diag)
    dual_ub(w) .= beta .* dual_ub(w) .+ alpha .* (x.xp_ur .* u_lower .+ dual_ub(x) .* u_diag)
    return
end

@inbounds function reduce_rhs!(
    xp_lr, wl, l_diag,
    xp_ur, wu, u_diag,
)
    xp_lr .-= wl ./ l_diag
    xp_ur .-= wu ./ u_diag
    return
end
function reduce_rhs!(kkt::AbstractKKTSystem, d::AbstractKKTVector)
    reduce_rhs!(
        d.xp_lr, dual_lb(d), kkt.l_diag,
        d.xp_ur, dual_ub(d), kkt.u_diag,
    )
end

# Finish
function finish_aug_solve!(kkt::AbstractKKTSystem, d::AbstractKKTVector)
    dlb = dual_lb(d)
    dub = dual_ub(d)
    dlb .= (.-dlb .+ kkt.l_lower .* d.xp_lr) ./ kkt.l_diag
    dub .= (  dub .- kkt.u_lower .* d.xp_ur) ./ kkt.u_diag
    return
end

function set_initial_bounds!(xl::AbstractVector{T}, xu::AbstractVector{T}, tol) where T
    # If `tol` is set to zero, keep the bounds unchanged.
    if tol > zero(T)
        map!(
            x->x - max(one(T), abs(x)) .* tol,
            xl, xl
        )
        map!(
            x->x + max(one(T), abs(x)) .* tol,
            xu, xu
        )
    end
end

function set_initial_rhs!(solver::AbstractMadNLPSolver{T}, kkt::AbstractKKTSystem) where T
    f = primal(_f(solver))
    zl = primal(_zl(solver))
    zu = primal(_zu(solver))
    px = primal(_p(solver))
    px .= .-f .+ zl .- zu
    fill!(dual(_p(solver)), zero(T))
    fill!(dual_lb(_p(solver)), zero(T))
    fill!(dual_ub(_p(solver)), zero(T))
    return
end

# Set ifr
function set_aug_rhs_ifr!(solver::AbstractMadNLPSolver{T}, kkt::AbstractKKTSystem, p0::AbstractKKTVector) where T
    fill!(primal(p0), zero(T))
    fill!(dual_lb(p0), zero(T))
    fill!(dual_ub(p0), zero(T))
    wy = dual(p0)
    wy .= .- _c(solver)
    return
end

function set_g_ifr!(solver::AbstractMadNLPSolver, g::AbstractArray)
    f = full(_f(solver))
    x = full(_x(solver))
    xl = full(_xl(solver))
    xu = full(_xu(solver))
    g .= f .- _mu(solver) ./ (x .- xl) .+ _mu(solver) ./ (xu .- x) .+ _jacl(solver)
end

# Finish RR
function finish_aug_solve_RR!(dpp, dnn, dzp, dzn, l, dl, pp, nn, zp, zn, mu_R, rho)
    dzp .= rho .- l .- dl .- zp
    dzn .= rho .+ l .+ dl .- zn
    dpp .= .- pp .+ mu_R ./zp .- (pp./zp) .* dzp
    dnn .= .- nn .+ mu_R ./zn .- (nn./zn) .* dzn
    return
end

# Kernel functions ---------------------------------------------------------
is_valid(val::R) where R <: Real = !(isnan(val) || isinf(val))
is_valid(vec::AbstractArray) = isempty(vec) ? true : mapreduce(is_valid, &, vec)

function _get_varphi(x1::T, x2::T, mu::T) where T
    x = x1 - x2
    if x < 0
        return T(Inf)
    else
        return -mu * log(x)
    end
end

function get_varphi(obj_val, x_lr, xl_r, xu_r, x_ur, mu)
    return obj_val + mapreduce(
        (x1,x2) -> _get_varphi(x1,x2,mu), +, x_lr, xl_r
    ) + mapreduce(
        (x1,x2) -> _get_varphi(x1,x2,mu), +, xu_r, x_ur
    )
end

@inline get_inf_pr(c::AbstractVector) = norm(c, Inf)

function get_inf_du(f, zl, zu, jacl, sd)
    return mapreduce((f,zl,zu,jacl) -> abs(f-zl+zu+jacl), max, f, zl, zu, jacl; init = zero(eltype(f))) / sd
end

function get_inf_compl(x_lr, xl_r, zl_r, xu_r, x_ur, zu_r, mu, sc)
    return max(
        mapreduce(
            (x_lr, xl_r, zl_r) -> abs((x_lr-xl_r)*zl_r-mu),
            max,
            x_lr, xl_r, zl_r;
            init = zero(eltype(x_lr))
        ),
        mapreduce(
            (xu_r, x_ur, zu_r) -> abs((xu_r-x_ur)*zu_r-mu),
            max,
            xu_r, x_ur, zu_r;
            init = zero(eltype(x_lr))
        )
    ) / sc
end

function get_varphi_d(
    f::AbstractVector{T},
    x::AbstractVector{T},
    xl::AbstractVector{T},
    xu::AbstractVector{T},
    dx::AbstractVector{T},
    mu,
) where T
    return mapreduce(
        (f,x,xl,xu,dx)-> (f - mu/(x-xl) + mu/(xu-x)) * dx,
        +,
        f, x, xl, xu, dx;
        init = zero(T)
    )
end

function get_alpha_max(
    x::AbstractVector{T},
    xl::AbstractVector{T},
    xu::AbstractVector{T},
    dx::AbstractVector{T},
    tau,
) where T
    return min(
        mapreduce(
            (x, xl, dx) -> dx < 0 ? (-x+xl)*tau/dx : T(Inf),
            min,

            x, xl, dx,
            init = one(T)
        ),
        mapreduce(
            (x, xu, dx) -> dx > 0 ? (-x+xu)*tau/dx : T(Inf),
            min,
            x, xu, dx,
            init = one(T)
        )
    )
end

function get_alpha_z(
    zl_r::AbstractVector{T},
    zu_r::AbstractVector{T},
    dzl::AbstractVector{T},
    dzu::AbstractVector{T},
    tau,
)  where T
    return min(
        mapreduce(
            (zl_r, dzl) -> dzl < 0 ? (-zl_r)*tau/dzl : T(Inf),
            min,
            zl_r, dzl,
            init = one(T)
        ),
        mapreduce(
            (zu_r, dzu) -> dzu < 0 ? (-zu_r)*tau/dzu : T(Inf),
            min,
            zu_r, dzu,
            init = one(T)
        )
    )
end

function get_obj_val_R(
    p::AbstractVector{T},
    n::AbstractVector{T},
    D_R::AbstractVector{T},
    x::AbstractVector{T},
    x_ref::AbstractVector{T},
    rho,
    zeta,
) where T
    return mapreduce(
        (p,n,D_R,x,x_ref) -> rho*(p+n) .+ zeta/2*D_R^2*(x-x_ref)^2,
        +,
        p,n,D_R,x,x_ref;
        init = zero(T)
    )
end

@inline get_theta(c) = norm(c, 1)

function get_theta_R(
    c::AbstractVector{T},
    p::AbstractVector{T},
    n::AbstractVector{T},
) where T
    return mapreduce(
        (c,p,n) -> abs(c-p+n),
        +,
        c,p,n;
        init = zero(T)
    )
end

function get_inf_pr_R(
    c::AbstractVector{T},
    p::AbstractVector{T},
    n::AbstractVector{T},
) where T
    return mapreduce(
        (c,p,n) -> abs(c-p+n),
        max,
        c,p,n;
        init = zero(T)
    )
end

function get_inf_du_R(
    f_R::AbstractVector{T},
    l::AbstractVector{T},
    zl::AbstractVector{T},
    zu::AbstractVector{T},
    jacl::AbstractVector{T},
    zp::AbstractVector{T},
    zn::AbstractVector{T},
    rho,
    sd,
)  where T
    return max(
        mapreduce(
            (f_R, zl, zu, jacl) -> abs(f_R-zl+zu+jacl),
            max,
            f_R, zl, zu, jacl;
            init = zero(T)
        ),
        mapreduce(
            (l, zp) -> abs(rho-l-zp),
            max,
            l, zp;
            init = zero(T)
        ),
        mapreduce(
            (l, zn) -> abs(rho+l-zn),
            max,
            l, zn;
            init = zero(T)
        )
    ) / sd
end

function get_inf_compl_R(
    x_lr::SubVector{T, VT, VI},
    xl_r,
    zl_r,
    xu_r,
    x_ur,
    zu_r,
    pp,
    zp,
    nn,
    zn,
    mu_R,
    sc
) where {T, VT <: AbstractVector{T}, VI}
    return max(
        mapreduce(
            (x_lr, xl_r, zl_r) -> abs((x_lr-xl_r)*zl_r-mu_R),
            max,
            x_lr, xl_r, zl_r;
            init = zero(T)
        ),
        mapreduce(
            (xu_r, x_ur, zu_r) -> abs((xu_r-x_ur)*zu_r-mu_R),
            max,
            xu_r, x_ur, zu_r;
            init = zero(T)
        ),
        mapreduce(
            (pp, zp) -> abs(pp*zp-mu_R),
            max,
            pp, zp;
            init = zero(T)
        ),
        mapreduce(
            (nn, zn) -> abs(nn*zn-mu_R),
            max,
            nn, zn;
            init = zero(T)
        ),
    ) / sc
end

function get_alpha_max_R(
    x::AbstractVector{T},
    xl::AbstractVector{T},
    xu::AbstractVector{T},
    dx::AbstractVector{T},
    pp::AbstractVector{T},
    dpp::AbstractVector{T},
    nn::AbstractVector{T},
    dnn::AbstractVector{T},
    tau_R,
) where T
    return min(
        mapreduce(
            (x,xl,xu,dx) -> if dx < 0
                (-x+xl)*tau_R/dx
            elseif dx > 0
                (-x+xu)*tau_R/dx
            else
                T(Inf)
            end,
            min,
            x,xl,xu,dx;
            init = one(T)
        ),
        mapreduce(
            (pp, dpp)-> if dpp < 0
                -pp*tau_R/dpp
            else
                T(Inf)
            end,
            min,
            pp, dpp;
            init = one(T)
        ),
        mapreduce(
            (nn, dnn)-> if dnn < 0
                -nn*tau_R/dnn
            else
                T(Inf)
            end,
            min,
            nn, dnn;
            init = one(T)
        )
    )
end

function get_alpha_z_R(
    zl_r::SubVector{T, VT, VI},
    zu_r,
    dzl,
    dzu,
    zp,
    dzp,
    zn,
    dzn,
    tau_R,
) where {T, VT <: AbstractVector{T}, VI}

    f(d,z) = d < 0 ? -z*tau_R/d : T(Inf)
    return min(
        mapreduce(
            f,
            min,
            dzl, zl_r;
            init = one(T)
        ),
        mapreduce(
            f,
            min,
            dzu, zu_r;
            init = one(T)
        ),
        mapreduce(
            f,
            min,
            dzp, zp;
            init = one(T)
        ),
        mapreduce(
            f,
            min,
            dzn, zn;
            init = one(T)
        )
    )
end

function get_varphi_R(
    obj_val,
    x_lr::SubVector{T, VT, VI},
    xl_r,
    xu_r,
    x_ur,
    pp,
    nn,
    mu_R,
)  where {T, VT <: AbstractVector{T}, VI}
    varphi_R = obj_val
    f1(x) = x < 0 ? T(Inf) : mu_R*log(x)
    function f2(x,y)
        d = x - y
        d < 0 ? T(Inf) : mu_R * log(d)
    end

    return obj_val - +(
        mapreduce(
            f2,
            +,
            x_lr, xl_r;
            init = zero(T)
        ),
        mapreduce(
            f2,
            +,
            xu_r, x_ur;
            init = zero(T)
        ),
        mapreduce(
            f1,
            +,
            pp;
            init = zero(T)
        ),
        mapreduce(
            f1,
            +,
            nn;
            init = zero(T)
        )
    )
end

function get_F(
    c::AbstractVector{T},
    f,
    zl,
    zu,
    jacl,
    x_lr,
    xl_r,
    zl_r,
    xu_r,
    x_ur,
    zu_r,
    mu,
) where T
    F1 = mapreduce(
        abs,
        +,
        c;
        init = zero(T)
    )
    F2 = mapreduce(
        (f,zl,zu,jacl) -> abs(f-zl+zu+jacl),
        +,
        f,zl,zu,jacl;
        init = zero(T)
    )
    F3 = mapreduce(
        (x_lr,xl_r,zl_r) -> (x_lr >= xl_r && zl_r >= 0) ? abs((x_lr-xl_r)*zl_r-mu) : T(Inf),
        +,
        x_lr,xl_r,zl_r;
        init = zero(T)
    )
    F4 = mapreduce(
        (xu_r,x_ur,zu_r) -> (xu_r >= x_ur && zu_r >= 0) ? abs((xu_r-xu_r)*zu_r-mu) : T(Inf),
        +,
        xu_r,xu_r,zu_r;
        init = zero(T)
    )
    return F1 + F2 + F3 + F4
end

function get_varphi_d_R(
    f_R::AbstractVector{T},
    x::AbstractVector{T},
    xl::AbstractVector{T},
    xu::AbstractVector{T},
    dx::AbstractVector{T},
    pp::AbstractVector{T},
    nn::AbstractVector{T},
    dpp::AbstractVector{T},
    dnn::AbstractVector{T},
    mu_R,
    rho,
) where T
    f(x,dx) = (rho - mu_R/x) * dx
    return +(
        mapreduce(
            (f_R, x, xl, xu, dx) -> (f_R - mu_R/(x-xl) + mu_R/(xu-x)) * dx,
            +,
            f_R, x, xl, xu, dx;
            init = zero(T)
        ),
        mapreduce(
            f,
            +,
            pp,dpp;
            init = zero(T)
        ),
        mapreduce(
            f,
            +,
            nn,dnn;
            init = zero(T)
        ),
    )
end

function _initialize_variables!(x::T, xl, xu, bound_push, bound_fac) where T
    if xl!=-T(Inf) && xu!=T(Inf)
        return min(
            xu-min(bound_push*max(1,abs(xu)), bound_fac*(xu-xl)),
            max(xl+min(bound_push*max(1,abs(xl)),bound_fac*(xu-xl)),x),
        )
    elseif xl!=-T(Inf) && xu==T(Inf)
        return max(xl+bound_push*max(1,abs(xl)), x)
    elseif xl==-T(Inf) && xu!=T(Inf)
        return min(xu-bound_push*max(1,abs(xu)), x)
    end
    return x
end

function initialize_variables!(x, xl, xu, bound_push, bound_fac)
    map!((x,l,u) -> _initialize_variables!(x,l,u, bound_push, bound_fac), x, x, xl, xu)
end

function adjust_boundary!(
    x_lr::AbstractVector{T},
    xl_r::AbstractVector{T},
    x_ur::AbstractVector{T},
    xu_r::AbstractVector{T},
    mu,
) where T
    c1 = eps(T)*mu
    c2 = eps(T)^(3/4)
    map!(
        (x_lr, xl_r) -> (x_lr-xl_r < c1) ? (xl_r - c2*max(1,abs(x_lr))) : xl_r,
        xl_r, x_lr, xl_r
    )
    map!(
        (xu_r, x_ur) -> (xu_r-x_ur < c1) ? (xu_r + c2*max(1,abs(x_ur))) : xu_r,
        xu_r, xu_r, x_ur
    )
end

function get_rel_search_norm(x::AbstractVector{T}, dx::AbstractVector{T}) where T
    return mapreduce(
        (x,dx) -> abs(dx) / (one(T) + abs(x)),
        max,
        x, dx
    )
end

# Utility functions
function get_sd(l, zl_r, zu_r, s_max)
    return max(
        s_max,
        (norm(l, 1)+norm(zl_r, 1)+norm(zu_r, 1)) / max(1, (length(l)+length(zl_r)+length(zu_r))),
    ) / s_max
end
function get_sc(zl_r, zu_r, s_max)
    return max(
        s_max,
        (norm(zl_r,1)+norm(zu_r,1)) / max(1,length(zl_r)+length(zu_r)),
    ) / s_max
end

function get_mu(
    mu::T,
    mu_min,
    mu_linear_decrease_factor,
    mu_superlinear_decrease_power,
    tol,
) where {T}
    # Warning: `a * tol` should be strictly less than 100 * mu_min, see issue #242
    a = min(T(99.0) * mu_min / tol, T(0.01))
    return max(
        mu_min,
        a * tol,
        min(mu_linear_decrease_factor*mu, mu^mu_superlinear_decrease_power),
    )
end

@inline get_tau(mu, tau_min) = max(tau_min, 1-mu)

function get_alpha_min(
    theta,
    varphi_d,
    theta_min,
    gamma_theta,
    gamma_phi,
    alpha_min_frac,
    del,
    s_theta,
    s_phi,
)
    if varphi_d<0
        if theta<=theta_min
            return alpha_min_frac*min(
                gamma_theta,gamma_phi*theta/(-varphi_d),
                del*theta^s_theta/(-varphi_d)^s_phi,
            )
        else
            return alpha_min_frac*min(
                gamma_theta,
                -gamma_phi*theta/varphi_d,
            )
        end
    else
        return alpha_min_frac*gamma_theta
    end
end

function is_switching(varphi_d, alpha, s_phi, del, theta, s_theta)
    return (varphi_d < 0) && (alpha*(-varphi_d)^s_phi > del*theta^s_theta)
end

function is_armijo(varphi_trial, varphi, eta_phi, alpha, varphi_d)
    return (varphi_trial <= varphi + eta_phi*alpha*varphi_d)
end

function is_sufficient_progress(theta_trial::T, theta, gamma_theta, varphi_trial, varphi, gamma_phi, has_constraints) where T
    (has_constraints && ((theta_trial<=(1-gamma_theta)*theta+10*eps(T)*abs(theta))) || ((varphi_trial<=varphi-gamma_phi*theta +10*eps(T)*abs(varphi))))
end

function augment_filter!(filter, theta, varphi, gamma_theta)
    push!(filter, ((1-gamma_theta)*theta, varphi-gamma_theta*theta))
end

function is_filter_acceptable(filter, theta, varphi)
    !isnan(theta) || return false
    !isinf(theta) || return false
    !isnan(varphi) || return false
    !isinf(varphi) || return false

    for (theta_F,varphi_F) in filter
        theta <= theta_F || varphi <= varphi_F || return false
    end
    return true
end

function is_barr_obj_rapid_increase(varphi::T, varphi_trial, obj_max_inc) where T
    return (varphi_trial >= varphi) && (log10(varphi_trial-varphi) > obj_max_inc + max(one(T), log10(abs(varphi))))
end

function reset_bound_dual!(
    z::AbstractVector{T},
    x::AbstractVector{T},
    mu,
    kappa_sigma,
) where T
    map!(
        (z, x) -> max(min(z, (kappa_sigma*mu)/x), (mu/kappa_sigma)/x),
        z, z, x
    )
    return
end

function reset_bound_dual!(
    z::AbstractVector{T},
    x1::AbstractVector{T},
    x2::AbstractVector{T},
    mu,
    kappa_sigma,
) where T
    map!(
        (z,x1,x2) -> max(min(z, (kappa_sigma*mu)/(x1-x2)), (mu/kappa_sigma)/(x1-x2)),
        z,z,x1,x2
    )
    return
end


function get_ftype(filter,theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
                   theta_min,obj_max_inc,gamma_theta,gamma_phi,has_constraints)
    is_filter_acceptable(filter,theta_trial,varphi_trial) || return " "
    !is_barr_obj_rapid_increase(varphi,varphi_trial,obj_max_inc) || return " "

    if theta <= theta_min && switching_condition
        armijo_condition && return "f"
    else
        is_sufficient_progress(
            theta_trial,theta,gamma_theta,varphi_trial,varphi,gamma_phi,has_constraints) && return "h"
    end

    return " "
end

# fixed variable treatment ----------------------------------------------------
function _get_fixed_variable_index(
    mat::SparseMatrixCSC{Tv,Ti1}, ind_fixed::Vector{Ti2}
) where {Tv,Ti1,Ti2}
    fixed_aug_index = Int[]
    for i in ind_fixed
        append!(fixed_aug_index,append!(collect(mat.colptr[i]+1:mat.colptr[i+1]-1)))
    end
    append!(fixed_aug_index,setdiff!(Base._findin(mat.rowval,ind_fixed),mat.colptr))

    return fixed_aug_index
end

function dual_inf_perturbation!(px, ind_llb, ind_uub, mu, kappa_d)
    px[ind_llb] .-= mu*kappa_d
    px[ind_uub] .+= mu*kappa_d
end

# Sections of the regular IPM algorithm:
function evaluate_termination_criteria!(solver::AbstractMadNLPSolver)
    @trace(_logger(solver),"Evaluating termination criteria.")
    _inf_total(solver) <= _opt(solver).tol && return SOLVE_SUCCEEDED
    _inf_total(solver) <= _opt(solver).acceptable_tol ?
        (_cnt(solver).acceptable_cnt < _opt(solver).acceptable_iter ?
        _cnt(solver).acceptable_cnt+=1 : return SOLVED_TO_ACCEPTABLE_LEVEL) : (_cnt(solver).acceptable_cnt = 0)
    _inf_total(solver) >= _opt(solver).diverging_iterates_tol && return DIVERGING_ITERATES
    _cnt(solver).k>=_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
    time()-_cnt(solver).start_time>=_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

    return REGULAR
end

function update_mu!(solver::AbstractMadNLPSolver{T}) where {T}
    @trace(_logger(solver),"Updating the barrier parameter.")
    while _mu(solver) != max(_opt(solver).mu_min,_opt(solver).tol/10) &&
        max(_inf_pr(solver),_inf_du(solver),_inf_compl_mu(solver)) <= _opt(solver).barrier_tol_factor*_mu(solver)
        mu_new = get_mu(_mu(solver),_opt(solver).mu_min,
                        _opt(solver).mu_linear_decrease_factor,_opt(solver).mu_superlinear_decrease_power,_opt(solver).tol)
        set_inf_compl_mu!(solver, get_inf_compl(_x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),_mu(solver),_sc(solver)))
        set_tau!(solver, get_tau(_mu(solver),_opt(solver).tau_min))
        set_mu!(solver, mu_new)
        empty!(_filter(solver))
        push!(_filter(solver),(_theta_max(solver),T(-Inf)))
    end
end

function compute_newton_step!(solver::AbstractMadNLPSolver)
    @trace(_logger(solver),"Computing the newton step.")
    if (_cnt(solver).k!=0 && !_opt(solver).hessian_constant)
        eval_lag_hess_wrapper!(solver, _kkt(solver), _x(solver), _y(solver))
    end
    set_aug_diagonal!(_kkt(solver),solver)
    set_aug_rhs!(solver, _kkt(solver), _c(solver))
    dual_inf_perturbation!(primal(_p(solver)),_ind_llb(solver),_ind_uub(solver),_mu(solver),_opt(solver).kappa_d)

    return inertia_correction!(_inertia_corrector(solver), solver) ? REGULAR : ROBUST
end

function line_search!(solver::AbstractMadNLPSolver)
    @trace(_logger(solver),"Backtracking line search initiated.")
    status = filter_line_search!(solver)
    return status
end

function update_variables!(solver::AbstractMadNLPSolver)
    @trace(_logger(solver),"Updating primal-dual variables.")
    copyto!(full(_x(solver)), full(_x_trial(solver)))
    copyto!(_c(solver), _c_trial(solver))
    set_obj_val!(solver, _obj_val_trial(solver))
    adjust_boundary!(_x_lr(solver),_xl_r(solver),_x_ur(solver),_xu_r(solver),_mu(solver))

    axpy!(_alpha(solver),dual(_d(solver)),_y(solver))

    _zl_r(solver) .+= _alpha_z(solver) .* dual_lb(_d(solver))
    _zu_r(solver) .+= _alpha_z(solver) .* dual_ub(_d(solver))
    reset_bound_dual!(
        primal(_zl(solver)),
        primal(_x(solver)),
        primal(_xl(solver)),
        _mu(solver),_opt(solver).kappa_sigma,
    )
    reset_bound_dual!(
        primal(_zu(solver)),
        primal(_xu(solver)),
        primal(_x(solver)),
        _mu(solver),_opt(solver).kappa_sigma,
    )
end

function eval_for_next_iter!(solver::AbstractMadNLPSolver{T}) where {T}
    if _cnt(solver).k!=0
        if !_opt(solver).jacobian_constant
            eval_jac_wrapper!(solver, _kkt(solver), _x(solver))
        end
        eval_grad_f_wrapper!(solver, _f(solver),_x(solver))
    end

    jtprod!(_jacl(solver), _kkt(solver), _y(solver))
    set_sd!(solver, get_sd(_y(solver),_zl_r(solver),_zu_r(solver),T(_opt(solver).s_max)))
    set_sc!(solver, get_sc(_zl_r(solver),_zu_r(solver),T(_opt(solver).s_max)))
    set_inf_pr!(solver, get_inf_pr(_c(solver)))
    set_inf_du!(solver, get_inf_du(
        full(_f(solver)),
        full(_zl(solver)),
        full(_zu(solver)),
        _jacl(solver),
        _sd(solver),
    ))
    set_inf_compl!(solver, _inf_compl(solver, mu=zero(T)))
    set_inf_compl_mu!(solver, _inf_compl(solver))
end

# Sections of the robust restorer IPM algorithm
function eval_for_next_iter_RR!(solver::AbstractMadNLPSolver{T}) where {T}
    RR = _RR(solver)
    if _cnt(solver).k!=0
        if !_opt(solver).jacobian_constant
            eval_jac_wrapper!(solver, _kkt(solver), _x(solver))
        end
        eval_grad_f_wrapper!(solver, _f(solver),_x(solver))
    end
    jtprod!(_jacl(solver), _kkt(solver), _y(solver))

    # evaluate termination criteria
    @trace(_logger(solver),"Evaluating restoration phase termination criteria.")

    sd = get_sd(_y(solver),_zl_r(solver),_zu_r(solver),_opt(solver).s_max)
    sc = get_sc(_zl_r(solver),_zu_r(solver),_opt(solver).s_max)
    set_inf_pr!(solver, get_inf_pr(_c(solver)))
    set_inf_du!(solver, get_inf_du(
        primal(_f(solver)),
        primal(_zl(solver)),
        primal(_zu(solver)),
        _jacl(solver),
        sd,
    ))
    set_inf_compl!(solver, get_inf_compl(_x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),zero(T),_sc(solver)))

    # Robust restoration phase error
    RR.inf_pr_R = get_inf_pr_R(_c(solver),RR.pp,RR.nn)
    RR.inf_du_R = get_inf_du_R(RR.f_R,_y(solver),primal(_zl(solver)),primal(_zu(solver)),_jacl(solver),RR.zp,RR.zn,_opt(solver).rho,_sd(solver))
    RR.inf_compl_R = get_inf_compl_R(
        _x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),RR.pp,RR.zp,RR.nn,RR.zn,zero(T),_sc(solver))
    RR.inf_compl_mu_R = get_inf_compl_R(
        _x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,_sc(solver))
end

function evaluate_termination_criteria_RR!(solver::AbstractMadNLPSolver)
    RR = _RR(solver)
    (max(RR.inf_pr_R,RR.inf_du_R,RR.inf_compl_R) <= _opt(solver).tol) && return INFEASIBLE_PROBLEM_DETECTED
    (_cnt(solver).k>=_opt(solver).max_iter) && return MAXIMUM_ITERATIONS_EXCEEDED
    (time()-_cnt(solver).start_time>=_opt(solver).max_wall_time) && return MAXIMUM_WALLTIME_EXCEEDED

    return ROBUST
end

function update_mu_RR!(solver::AbstractMadNLPSolver)
    RR = _RR(solver)
    @trace(_logger(solver),"Updating restoration phase barrier parameter.")
    while RR.mu_R >= _opt(solver).mu_min &&
        max(RR.inf_pr_R,RR.inf_du_R,RR.inf_compl_mu_R) <= _opt(solver).barrier_tol_factor*RR.mu_R
        RR.mu_R = get_mu(RR.mu_R,_opt(solver).mu_min,
                         _opt(solver).mu_linear_decrease_factor,_opt(solver).mu_superlinear_decrease_power,_opt(solver).tol)
        RR.inf_compl_mu_R = get_inf_compl_R(
            _x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,_sc(solver))
        RR.tau_R= max(_opt(solver).tau_min,1-RR.mu_R)
        RR.zeta = sqrt(RR.mu_R)

        empty!(RR.filter)
        push!(RR.filter,(_theta_max(solver),-Inf))
    end
end

function compute_newton_step_RR!(solver::AbstractMadNLPSolver)
    RR = _RR(solver)
    if !_opt(solver).hessian_constant
        eval_lag_hess_wrapper!(solver, _kkt(solver), _x(solver), _y(solver); is_resto=true)
    end
    set_aug_RR!(_kkt(solver), solver, RR)

    # without inertia correction,
    @trace(_logger(solver),"Solving restoration phase primal-dual system.")
    set_aug_rhs_RR!(solver, _kkt(solver), RR, _opt(solver).rho)

    inertia_correction!(_inertia_corrector(solver), solver) || return RESTORATION_FAILED

    finish_aug_solve_RR!(
        RR.dpp,RR.dnn,RR.dzp,RR.dzn,_y(solver),dual(_d(solver)),
        RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,_opt(solver).rho
    )
    return ROBUST
end

function line_search_RR!(solver::AbstractMadNLPSolver)
    @trace(_logger(solver),"Backtracking line search initiated.")
    status = filter_line_search_RR!(solver)
    return status
end

function update_variables_RR!(solver::AbstractMadNLPSolver)
    RR = _RR(solver)
    @trace(_logger(solver),"Updating primal-dual variables.")
    copyto!(full(_x(solver)), full(_x_trial(solver)))
    copyto!(_c(solver), _c_trial(solver))
    copyto!(RR.pp, RR.pp_trial)
    copyto!(RR.nn, RR.nn_trial)

    RR.obj_val_R=RR.obj_val_R_trial
    set_f_RR!(solver,RR)

    axpy!(_alpha(solver), dual(_d(solver)), _y(solver))
    axpy!(_alpha_z(solver), RR.dzp,RR.zp)
    axpy!(_alpha_z(solver), RR.dzn,RR.zn)

    _zl_r(solver) .+= _alpha_z(solver) .* dual_lb(_d(solver))
    _zu_r(solver) .+= _alpha_z(solver) .* dual_ub(_d(solver))

    reset_bound_dual!(
        primal(_zl(solver)),
        primal(_x(solver)),
        primal(_xl(solver)),
        RR.mu_R, _opt(solver).kappa_sigma,
    )
    reset_bound_dual!(
        primal(_zu(solver)),
        primal(_xu(solver)),
        primal(_x(solver)),
        RR.mu_R, _opt(solver).kappa_sigma,
    )
    reset_bound_dual!(RR.zp,RR.pp,RR.mu_R,_opt(solver).kappa_sigma)
    reset_bound_dual!(RR.zn,RR.nn,RR.mu_R,_opt(solver).kappa_sigma)

    adjust_boundary!(_x_lr(solver),_xl_r(solver),_x_ur(solver),_xu_r(solver),_mu(solver))
end


function check_restoration_successful!(solver::AbstractMadNLPSolver)
    RR = _RR(solver)
    @trace(_logger(solver),"Checking if going back to regular phase.")
    set_obj_val!(solver, eval_f_wrapper(solver, _x(solver)))
    eval_grad_f_wrapper!(solver, _f(solver), _x(solver))
    theta = get_theta(_c(solver))
    varphi= get_varphi(_obj_val(solver),_x_lr(solver),_xl_r(solver),_xu_r(solver),_x_ur(solver),_mu(solver))

    if is_filter_acceptable(_filter(solver),theta,varphi) &&
        theta <= _opt(solver).required_infeasibility_reduction * RR.theta_ref
        return REGULAR
    else
        return ROBUST
    end
end

function return_from_restoration!(solver::AbstractMadNLPSolver{T}) where {T}
    RR = _RR(solver)
    @trace(_logger(solver),"Going back to the regular phase.")
    set_initial_rhs!(solver, _kkt(solver))
    initialize!(_kkt(solver))

    factorize_wrapper!(solver)
    solve_refine_wrapper!(
        _d(solver), solver, _p(solver), __w4(solver)
    )
    if norm(dual(_d(solver)), Inf)>_opt(solver).constr_mult_init_max
        fill!(_y(solver), zero(T))
    else
        copyto!(_y(solver), dual(_d(solver)))
    end
end
