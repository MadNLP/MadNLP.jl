
# KKT system updates -------------------------------------------------------
# Set diagonal
function set_aug_diagonal!(kkt::AbstractKKTSystem{T}, solver::AbstractMadNLPSolver{T}) where T
    x = full(get_x(solver))
    xl = full(get_xl(solver))
    xu = full(get_xu(solver))
    zl = full(get_zl(solver))
    zu = full(get_zu(solver))

    fill!(kkt.reg, zero(T))
    fill!(kkt.du_diag, zero(T))
    kkt.l_diag .= get_xl_r(solver) .- get_x_lr(solver)   # (Xˡ - X)
    kkt.u_diag .= get_x_ur(solver) .- get_xu_r(solver)   # (X - Xᵘ)
    copyto!(kkt.l_lower, get_zl_r(solver))
    copyto!(kkt.u_lower, get_zu_r(solver))

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
    kkt.l_diag .= get_x_lr(solver) .- get_xl_r(solver)   # (X - Xˡ)
    kkt.u_diag .= get_xu_r(solver) .- get_x_ur(solver)   # (Xᵘ - X)
    copyto!(kkt.l_lower, get_zl_r(solver))
    copyto!(kkt.u_lower, get_zu_r(solver))
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
    x = full(get_x(solver))
    xl = full(get_xl(solver))
    xu = full(get_xu(solver))
    zl = full(get_zl(solver))
    zu = full(get_zu(solver))
    kkt.reg .= RR.zeta .* RR.D_R .^ 2
    kkt.du_diag .= .- RR.pp ./ RR.zp .- RR.nn ./ RR.zn
    copyto!(kkt.l_lower, get_zl_r(solver))
    copyto!(kkt.u_lower, get_zu_r(solver))
    kkt.l_diag .= get_xl_r(solver) .- get_x_lr(solver)
    kkt.u_diag .= get_x_ur(solver) .- get_xu_r(solver)

    _set_aug_diagonal!(kkt)
    return
end

function set_aug_RR!(kkt::ScaledSparseKKTSystem, solver::AbstractMadNLPSolver, RR::RobustRestorer)
    x = full(get_x(solver))
    xl = full(get_xl(solver))
    xu = full(get_xu(solver))
    zl = full(get_zl(solver))
    zu = full(get_zu(solver))
    kkt.reg .= RR.zeta .* RR.D_R .^ 2
    kkt.du_diag .= .- RR.pp ./ RR.zp .- RR.nn ./ RR.zn
    copyto!(kkt.l_lower, get_zl_r(solver))
    copyto!(kkt.u_lower, get_zu_r(solver))
    kkt.l_diag .= get_x_lr(solver) .- get_xl_r(solver)
    kkt.u_diag .= get_xu_r(solver) .- get_x_ur(solver)

    _set_aug_diagonal!(kkt)
    return
end

function set_f_RR!(solver::AbstractMadNLPSolver, RR::RobustRestorer)
    x = full(get_x(solver))
    RR.f_R .= RR.zeta .* RR.D_R .^ 2 .* (x .- RR.x_ref)
    return
end

# Set RHS
function set_aug_rhs!(solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, c::AbstractVector, mu)
    px = primal(get_p(solver))
    x = primal(get_x(solver))
    f = primal(get_f(solver))
    xl = primal(get_xl(solver))
    xu = primal(get_xu(solver))
    zl = full(get_zl(solver))
    zu = full(get_zu(solver))
    py = dual(get_p(solver))
    pzl = dual_lb(get_p(solver))
    pzu = dual_ub(get_p(solver))

    px .= .-f .+ zl .- zu .- get_jacl(solver)
    py .= .-c
    pzl .= (get_xl_r(solver) .- get_x_lr(solver)) .* get_zl_r(solver) .+ mu
    pzu .= (get_xu_r(solver) .- get_x_ur(solver)) .* get_zu_r(solver) .- mu
    return
end

# Set RHS RR
function set_aug_rhs_RR!(
    solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, RR::RobustRestorer, rho,
)
    x = full(get_x(solver))
    xl = full(get_xl(solver))
    xu = full(get_xu(solver))
    zl = full(get_zl(solver))
    zu = full(get_zu(solver))

    px = primal(get_p(solver))
    py = dual(get_p(solver))
    pzl = dual_lb(get_p(solver))
    pzu = dual_ub(get_p(solver))

    mu = RR.mu_R

    px .= .- RR.f_R .+ zl .- zu .- get_jacl(solver)
    py .= .- get_c(solver) .+ RR.pp .- RR.nn .+
        (mu .- (rho .- get_y(solver)) .* RR.pp) ./ RR.zp .-
        (mu .- (rho .+ get_y(solver)) .* RR.nn) ./ RR.zn

    pzl .= (get_xl_r(solver) .- get_x_lr(solver)) .* get_zl_r(solver) .+ mu
    pzu .= (get_xu_r(solver) .- get_x_ur(solver)) .* get_zu_r(solver) .- mu

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
    f = primal(get_f(solver))
    zl = primal(get_zl(solver))
    zu = primal(get_zu(solver))
    px = primal(get_p(solver))
    px .= .-f .+ zl .- zu
    fill!(dual(get_p(solver)), zero(T))
    fill!(dual_lb(get_p(solver)), zero(T))
    fill!(dual_ub(get_p(solver)), zero(T))
    return
end

# Set ifr
function set_aug_rhs_ifr!(solver::AbstractMadNLPSolver{T}, kkt::AbstractKKTSystem, p0::AbstractKKTVector) where T
    fill!(primal(p0), zero(T))
    fill!(dual_lb(p0), zero(T))
    fill!(dual_ub(p0), zero(T))
    wy = dual(p0)
    wy .= .- get_c(solver)
    return
end

function set_g_ifr!(solver::AbstractMadNLPSolver, g::AbstractArray)
    f = full(get_f(solver))
    x = full(get_x(solver))
    xl = full(get_xl(solver))
    xu = full(get_xu(solver))
    g .= f .- get_mu(solver) ./ (x .- xl) .+ get_mu(solver) ./ (xu .- x) .+ get_jacl(solver)
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

function get_average_complementarity(x_lr, xl_r, zl_r, x_ur, xu_r, zu_r)
    n_lb, n_ub = length(x_lr), length(x_ur)
    # If the problem is unconstrained, average complementarity is 0
    if n_lb + n_ub == 0
        return 0.0
    end
    cc_lb = dot(x_lr, zl_r) - dot(xl_r, zl_r)
    cc_ub = dot(xu_r, zu_r) - dot(x_ur, zu_r)
    return (cc_lb + cc_ub) / (n_lb + n_ub)
end
function get_average_complementarity(solver::AbstractMadNLPSolver)
    return get_average_complementarity(
        solver.x_lr, solver.xl_r, solver.zl_r,
        solver.x_ur, solver.xu_r, solver.zu_r,
    )
end

function get_min_complementarity(x_lr::AbstractVector{T}, xl_r::AbstractVector{T}, zl_r::AbstractVector{T},
                                 x_ur::AbstractVector{T}, xu_r::AbstractVector{T}, zu_r::AbstractVector{T}) where T
    cc_lb = mapreduce((x_l, xl, zl) -> (x_l-xl)*zl, min, x_lr, xl_r, zl_r, init=T(Inf))
    cc_ub = mapreduce((x_u, xu, zu) -> (xu-x_u)*zu, min, x_ur, xu_r, zu_r, init=T(Inf))
    return min(cc_lb,cc_ub)
end

function get_min_complementarity(solver::AbstractMadNLPSolver)
    return get_min_complementarity(
        solver.x_lr, solver.xl_r, solver.zl_r,
        solver.x_ur, solver.xu_r, solver.zu_r,
    )
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
function eval_for_next_iter!(solver::AbstractMadNLPSolver{T}) where {T}
    if get_cnt(solver).k!=0
        if !get_opt(solver).jacobian_constant
            eval_jac_wrapper!(solver, get_kkt(solver), get_x(solver))
        end
        eval_grad_f_wrapper!(solver, get_f(solver),get_x(solver))
    end

    jtprod!(get_jacl(solver), get_kkt(solver), get_y(solver))
    set_sd!(solver, get_sd(get_y(solver),get_zl_r(solver),get_zu_r(solver),T(get_opt(solver).s_max)))
    set_sc!(solver, get_sc(get_zl_r(solver),get_zu_r(solver),T(get_opt(solver).s_max)))
    set_inf_pr!(solver, get_inf_pr(get_c(solver)))
    set_inf_du!(solver, get_inf_du(
        full(get_f(solver)),
        full(get_zl(solver)),
        full(get_zu(solver)),
        get_jacl(solver),
        get_sd(solver),
    ))
    set_inf_compl!(solver, get_inf_compl(solver, get_sc(solver); mu=zero(T)))
    set_inf_compl_mu!(solver, get_inf_compl(solver))
end

function evaluate_termination_criteria!(solver::AbstractMadNLPSolver)
    @trace(get_logger(solver),"Evaluating termination criteria.")
    get_inf_total(solver) <= get_opt(solver).tol && return SOLVE_SUCCEEDED
    get_inf_total(solver) <= get_opt(solver).acceptable_tol ?
        (get_cnt(solver).acceptable_cnt < get_opt(solver).acceptable_iter ?
        get_cnt(solver).acceptable_cnt+=1 : return SOLVED_TO_ACCEPTABLE_LEVEL) : (get_cnt(solver).acceptable_cnt = 0)
    get_inf_total(solver) >= get_opt(solver).diverging_iterates_tol && return DIVERGING_ITERATES
    get_cnt(solver).k>=get_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
    time()-get_cnt(solver).start_time>=get_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

    return REGULAR
end

function evaluate_hessian!(solver::AbstractMadNLPSolver)
    @trace(get_logger(solver),"Evaluating the Hessian of the Lagrangian.")
    if (get_cnt(solver).k!=0 && !get_opt(solver).hessian_constant)
        eval_lag_hess_wrapper!(solver, get_kkt(solver), get_x(solver), get_y(solver))
    end
end

function update_homotopy_parameters!(solver::AbstractMadNLPSolver)
    @trace(get_logger(solver),"Updating the barrier parameter.")
    update_barrier!(get_opt(solver).barrier, solver, get_sc(solver))
end

function compute_newton_step!(solver::AbstractMadNLPSolver)
    @trace(get_logger(solver),"Computing the newton step.")
    set_aug_diagonal!(get_kkt(solver),solver)
    set_aug_rhs!(solver, get_kkt(solver), get_c(solver), get_mu(solver))
    dual_inf_perturbation!(primal(get_p(solver)),get_ind_llb(solver),get_ind_uub(solver),get_mu(solver),get_opt(solver).kappa_d)

    return inertia_correction!(get_inertia_corrector(solver), solver) ? REGULAR : ROBUST
end

function line_search!(solver::AbstractMadNLPSolver)
    @trace(get_logger(solver),"Backtracking line search initiated.")
    status = filter_line_search!(solver)
    return status
end

function update_variables!(solver::AbstractMadNLPSolver)
    @trace(get_logger(solver),"Updating primal-dual variables.")
    copyto!(full(get_x(solver)), full(get_x_trial(solver)))
    copyto!(get_c(solver), get_c_trial(solver))
    set_obj_val!(solver, get_obj_val_trial(solver))
    adjust_boundary!(get_x_lr(solver),get_xl_r(solver),get_x_ur(solver),get_xu_r(solver),get_mu(solver))

    axpy!(get_alpha(solver),dual(get_d(solver)),get_y(solver))

    get_zl_r(solver) .+= get_alpha_z(solver) .* dual_lb(get_d(solver))
    get_zu_r(solver) .+= get_alpha_z(solver) .* dual_ub(get_d(solver))
    reset_bound_dual!(
        primal(get_zl(solver)),
        primal(get_x(solver)),
        primal(get_xl(solver)),
        get_mu(solver),get_opt(solver).kappa_sigma,
    )
    reset_bound_dual!(
        primal(get_zu(solver)),
        primal(get_xu(solver)),
        primal(get_x(solver)),
        get_mu(solver),get_opt(solver).kappa_sigma,
    )
end

# Sections of the robust restorer IPM algorithm
function eval_for_next_iter_RR!(solver::AbstractMadNLPSolver{T}) where {T}
    RR = get_RR(solver)
    if get_cnt(solver).k!=0
        if !get_opt(solver).jacobian_constant
            eval_jac_wrapper!(solver, get_kkt(solver), get_x(solver))
        end
        eval_grad_f_wrapper!(solver, get_f(solver),get_x(solver))
    end
    jtprod!(get_jacl(solver), get_kkt(solver), get_y(solver))

    # evaluate termination criteria
    @trace(get_logger(solver),"Evaluating restoration phase termination criteria.")

    set_sd!(solver, get_sd(get_y(solver),get_zl_r(solver),get_zu_r(solver),get_opt(solver).s_max))
    set_sc!(solver, get_sc(get_zl_r(solver),get_zu_r(solver),get_opt(solver).s_max))
    set_inf_pr!(solver, get_inf_pr(get_c(solver)))
    set_inf_du!(solver, get_inf_du(
        primal(get_f(solver)),
        primal(get_zl(solver)),
        primal(get_zu(solver)),
        get_jacl(solver),
        get_sd(solver),
    ))
    set_inf_compl!(solver, get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),zero(T),get_sc(solver)))

    # Robust restoration phase error
    RR.inf_pr_R = get_inf_pr_R(get_c(solver),RR.pp,RR.nn)
    RR.inf_du_R = get_inf_du_R(RR.f_R,get_y(solver),primal(get_zl(solver)),primal(get_zu(solver)),get_jacl(solver),RR.zp,RR.zn,get_opt(solver).rho,get_sd(solver))
    RR.inf_compl_R = get_inf_compl_R(
        get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),RR.pp,RR.zp,RR.nn,RR.zn,zero(T),get_sc(solver))
    RR.inf_compl_mu_R = get_inf_compl_R(
        get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,get_sc(solver))
end

function evaluate_termination_criteria_RR!(solver::AbstractMadNLPSolver)
    RR = get_RR(solver)
    (max(RR.inf_pr_R,RR.inf_du_R,RR.inf_compl_R) <= get_opt(solver).tol) && return INFEASIBLE_PROBLEM_DETECTED
    (get_cnt(solver).k>=get_opt(solver).max_iter) && return MAXIMUM_ITERATIONS_EXCEEDED
    (time()-get_cnt(solver).start_time>=get_opt(solver).max_wall_time) && return MAXIMUM_WALLTIME_EXCEEDED

    return ROBUST
end

function update_homotopy_parameters_RR!(solver::AbstractMadNLPSolver)
    RR = get_RR(solver)
    @trace(get_logger(solver),"Updating restoration phase barrier parameter.")
    _update_monotone_RR!(get_opt(solver).barrier, solver, get_sc(solver))
end

function compute_newton_step_RR!(solver::AbstractMadNLPSolver)
    RR = get_RR(solver)
    if !get_opt(solver).hessian_constant
        eval_lag_hess_wrapper!(solver, get_kkt(solver), get_x(solver), get_y(solver); is_resto=true)
    end
    set_aug_RR!(get_kkt(solver), solver, RR)

    # without inertia correction,
    @trace(get_logger(solver),"Solving restoration phase primal-dual system.")
    set_aug_rhs_RR!(solver, get_kkt(solver), RR, get_opt(solver).rho)

    inertia_correction!(get_inertia_corrector(solver), solver) || return RESTORATION_FAILED

    finish_aug_solve_RR!(
        RR.dpp,RR.dnn,RR.dzp,RR.dzn,get_y(solver),dual(get_d(solver)),
        RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,get_opt(solver).rho
    )
    return ROBUST
end

function line_search_RR!(solver::AbstractMadNLPSolver)
    @trace(get_logger(solver),"Backtracking line search initiated.")
    status = filter_line_search_RR!(solver)
    return status
end

function update_variables_RR!(solver::AbstractMadNLPSolver)
    RR = get_RR(solver)
    @trace(get_logger(solver),"Updating primal-dual variables.")
    copyto!(full(get_x(solver)), full(get_x_trial(solver)))
    copyto!(get_c(solver), get_c_trial(solver))
    copyto!(RR.pp, RR.pp_trial)
    copyto!(RR.nn, RR.nn_trial)

    RR.obj_val_R=RR.obj_val_R_trial
    set_f_RR!(solver,RR)

    axpy!(get_alpha(solver), dual(get_d(solver)), get_y(solver))
    axpy!(get_alpha_z(solver), RR.dzp,RR.zp)
    axpy!(get_alpha_z(solver), RR.dzn,RR.zn)

    get_zl_r(solver) .+= get_alpha_z(solver) .* dual_lb(get_d(solver))
    get_zu_r(solver) .+= get_alpha_z(solver) .* dual_ub(get_d(solver))

    reset_bound_dual!(
        primal(get_zl(solver)),
        primal(get_x(solver)),
        primal(get_xl(solver)),
        RR.mu_R, get_opt(solver).kappa_sigma,
    )
    reset_bound_dual!(
        primal(get_zu(solver)),
        primal(get_xu(solver)),
        primal(get_x(solver)),
        RR.mu_R, get_opt(solver).kappa_sigma,
    )
    reset_bound_dual!(RR.zp,RR.pp,RR.mu_R,get_opt(solver).kappa_sigma)
    reset_bound_dual!(RR.zn,RR.nn,RR.mu_R,get_opt(solver).kappa_sigma)

    adjust_boundary!(get_x_lr(solver),get_xl_r(solver),get_x_ur(solver),get_xu_r(solver),get_mu(solver))
end

function check_restoration_successful!(solver::AbstractMadNLPSolver)
    RR = get_RR(solver)
    @trace(get_logger(solver),"Checking if going back to regular phase.")
    set_obj_val!(solver, eval_f_wrapper(solver, get_x(solver)))
    eval_grad_f_wrapper!(solver, get_f(solver), get_x(solver))
    theta = get_theta(get_c(solver))
    varphi= get_varphi(get_obj_val(solver),get_x_lr(solver),get_xl_r(solver),get_xu_r(solver),get_x_ur(solver),get_mu(solver))

    if is_filter_acceptable(get_filter(solver),theta,varphi) &&
        theta <= get_opt(solver).required_infeasibility_reduction * RR.theta_ref
        return REGULAR
    else
        return ROBUST
    end
end

function return_from_restoration!(solver::AbstractMadNLPSolver{T}) where {T}
    RR = get_RR(solver)
    @trace(get_logger(solver),"Going back to the regular phase.")
    set_initial_rhs!(solver, get_kkt(solver))
    initialize!(get_kkt(solver))

    factorize_wrapper!(solver)
    solve_refine_wrapper!(
        get_d(solver), solver, get_p(solver), get__w4(solver)
    )
    if norm(dual(get_d(solver)), T(Inf))>get_opt(solver).constr_mult_init_max
        fill!(get_y(solver), zero(T))
    else
        copyto!(get_y(solver), dual(get_d(solver)))
    end
end

# Sections of `restore!` heuristic.
function initialize_restore!(solver::AbstractMadNLPSolver{T}) where T
    set_del_w!(solver, zero(T))
    # Backup the previous primal iterate
    copyto!(primal(get__w1(solver)), full(get_x(solver)))
    copyto!(dual(get__w1(solver)), get_y(solver))
    copyto!(dual(get__w2(solver)), get_c(solver))
    set_alpha_z!(solver, zero(T))
    set_ftype!(solver, "R")

    return nothing
end

function take_ftb_step!(solver::AbstractMadNLPSolver{T}) where T
    alpha_max = get_alpha_max(
        primal(get_x(solver)),
        primal(get_xl(solver)),
        primal(get_xu(solver)),
        primal(get_d(solver)),
        get_tau(solver),
    )
    set_alpha!(solver,min(
        alpha_max,
        get_alpha_z(get_zl_r(solver),get_zu_r(solver),dual_lb(get_d(solver)),dual_ub(get_d(solver)),get_tau(solver)),
    ))

    axpy!(get_alpha(solver), primal(get_d(solver)), full(get_x(solver)))
    axpy!(get_alpha(solver), dual(get_d(solver)), get_y(solver))
    get_zl_r(solver) .+= get_alpha(solver) .* dual_lb(get_d(solver))
    get_zu_r(solver) .+= get_alpha(solver) .* dual_ub(get_d(solver))

    return nothing
end

function evaluate_for_sufficient_decrease_restore!(solver::AbstractMadNLPSolver{T}) where T
    eval_cons_wrapper!(solver,get_c(solver),get_x(solver))
    eval_grad_f_wrapper!(solver,get_f(solver),get_x(solver))
    set_obj_val!(solver, eval_f_wrapper(solver,get_x(solver)))

    if !get_opt(solver).jacobian_constant
        eval_jac_wrapper!(solver,get_kkt(solver),get_x(solver))
    end

    jtprod!(get_jacl(solver),get_kkt(solver),get_y(solver))

    return nothing
end

function backtrack_restore!(solver::AbstractMadNLPSolver{T}) where T
    copyto!(primal(get_x(solver)), primal(get__w1(solver)))
    copyto!(get_y(solver), dual(get__w1(solver)))
    copyto!(get_c(solver), dual(get__w2(solver))) # backup the previous primal iterate

    return nothing
end

function evaluate_termination_criteria_restore!(solver::AbstractMadNLPSolver{T}) where T
    theta = get_theta(get_c(solver))
    varphi= get_varphi(get_obj_val(solver),get_x_lr(solver),get_xl_r(solver),get_xu_r(solver),get_x_ur(solver),get_mu(solver))

    get_cnt(solver).k+=1

    is_filter_acceptable(get_filter(solver),theta,varphi) ? (return REGULAR) : (get_cnt(solver).t+=1)
    get_cnt(solver).k>=get_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
    time()-get_cnt(solver).start_time>=get_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

    return RESTORE
end

function evaluate_for_next_iter_restore!(solver::AbstractMadNLPSolver{T}) where T
    set_sd!(solver, get_sd(get_y(solver),get_zl_r(solver),get_zu_r(solver),get_opt(solver).s_max))
    set_sc!(solver, get_sc(get_zl_r(solver),get_zu_r(solver),get_opt(solver).s_max))
    set_inf_pr!(solver, get_inf_pr(get_c(solver)))
    set_inf_du!(solver, get_inf_du(
        primal(get_f(solver)),
        primal(get_zl(solver)),
        primal(get_zu(solver)),
        get_jacl(solver),
        get_sd(solver),
    ))

    set_inf_compl!(solver, get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),zero(T),get_sc(solver)))
    set_inf_compl_mu!(solver, get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),get_mu(solver),get_sc(solver)))

    return nothing
end

function compute_newton_step_restore!(solver::AbstractMadNLPSolver{T}) where T
    if !get_opt(solver).hessian_constant
        eval_lag_hess_wrapper!(solver,get_kkt(solver),get_x(solver),get_y(solver))
    end

    set_aug_diagonal!(get_kkt(solver),solver)
    set_aug_rhs!(solver, get_kkt(solver), get_c(solver), get_mu(solver))

    dual_inf_perturbation!(primal(get_p(solver)),get_ind_llb(solver),get_ind_uub(solver),get_mu(solver),get_opt(solver).kappa_d)
    factorize_wrapper!(solver)
    solve_refine_wrapper!(
        get_d(solver), solver, get_p(solver), get__w4(solver)
    )
end
