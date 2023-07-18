
# KKT system updates -------------------------------------------------------
# Set diagonal
# temporaily commented out
# function set_aug_diagonal!(kkt::AbstractKKTSystem, solver::MadNLPSolver{T}) where T
#     x = full(solver.x)
#     xl = full(solver.xl)
#     xu = full(solver.xu)
#     zl = full(solver.zl)
#     zu = full(solver.zu)
#     @inbounds @simd for i in eachindex(kkt.pr_diag)
#         kkt.pr_diag[i] = zl[i] /(x[i] - xl[i])
#         kkt.pr_diag[i] += zu[i] /(xu[i] - x[i])
#     end
#     fill!(kkt.du_diag, zero(T))
#     @inbounds @simd for i in eachindex(kkt.l_lower)
#         kkt.l_lower[i] = solver.zl_r[i]
#         kkt.l_diag[i]  = solver.x_lr[i] - solver.xl_r[i]
#     end
#     @inbounds @simd for i in eachindex(kkt.u_lower)
#         kkt.u_lower[i] = solver.zu_r[i]
#         kkt.u_diag[i] = solver.x_ur[i] - solver.xu_r[i]
#     end

#     return
# end
function set_aug_diagonal!(kkt::AbstractKKTSystem{T}, solver::MadNLPSolver{T}) where T
    x = full(solver.x)
    xl = full(solver.xl)
    xu = full(solver.xu)
    zl = full(solver.zl)
    zu = full(solver.zu)
    
    kkt.pr_diag .= zl ./(x .- xl) .+ zu ./(xu .- x)
    fill!(kkt.du_diag, zero(T))
    kkt.l_diag .= solver.xl_r .- solver.x_lr
    kkt.u_diag .= solver.x_ur .- solver.xu_r
    kkt.l_lower .= solver.zl_r
    kkt.u_lower .= solver.zu_r

    return
end

function set_aug_diagonal!(kkt::SparseUnreducedKKTSystem{T}, solver::MadNLPSolver{T}) where T
    fill!(kkt.pr_diag, zero(T))
    fill!(kkt.du_diag, zero(T))
    kkt.l_diag .= solver.xl_r .- solver.x_lr
    kkt.u_diag .= solver.x_ur .- solver.xu_r
    kkt.l_lower .= solver.zl_r
    kkt.u_lower .= solver.zu_r
    kkt.l_lower_aug .= sqrt.(kkt.l_lower)
    kkt.u_lower_aug .= sqrt.(kkt.u_lower)
    return
end

# Robust restoration
function set_aug_RR!(kkt::AbstractKKTSystem, solver::MadNLPSolver, RR::RobustRestorer)
    x = full(solver.x)
    xl = full(solver.xl)
    xu = full(solver.xu)
    zl = full(solver.zl)
    zu = full(solver.zu)
    kkt.pr_diag .= zl ./(x .- xl) .+ zu ./(xu .- x) .+ RR.zeta .* RR.D_R .^ 2
    kkt.du_diag .= .- RR.pp ./ RR.zp .- RR.nn ./ RR.zn
    kkt.l_diag  .= solver.xl_r .- solver.x_lr
    kkt.u_diag .= solver.x_ur .- solver.xu_r
    kkt.l_lower_aug .= sqrt.(kkt.l_lower)
    kkt.u_lower_aug .= sqrt.(kkt.u_lower)

    return
end
function set_aug_RR!(kkt::SparseUnreducedKKTSystem, solver::MadNLPSolver, RR::RobustRestorer)
    kkt.pr_diag .= RR.zeta .* RR.D_R.^2
    kkt.du_diag .= .-RR.pp ./ RR.zp .- RR.nn ./ RR.zn
    kkt.l_diag  .= solver.xl_r .- solver.x_lr
    kkt.u_diag .= solver.x_ur .- solver.xu_r
    kkt.l_lower .= solver.zl_r
    kkt.u_lower .= solver.zu_r
    kkt.l_lower_aug .= sqrt.(kkt.l_lower)
    kkt.u_lower_aug .= sqrt.(kkt.u_lower)
    return
end
function set_f_RR!(solver::MadNLPSolver, RR::RobustRestorer)
    x = full(solver.x)
    RR.f_R .= RR.zeta .* RR.D_R .^ 2 .* (x .- RR.x_ref)
end


# Set RHS
function set_aug_rhs!(solver::MadNLPSolver, kkt::AbstractKKTSystem, c)
    px = primal(solver.p)
    x = primal(solver.x)
    f = primal(solver.f)
    xl = primal(solver.xl)
    xu = primal(solver.xu)
    zl = full(solver.zl)
    zu = full(solver.zu)
    py = dual(solver.p)
    pzl = dual_lb(solver.p)
    pzu = dual_ub(solver.p)

    px .= .-f .+ zl .- zu .- solver.jacl
    py .= .-c
    pzl .= (solver.xl_r .- solver.x_lr) .* solver.zl_r .+ solver.mu 
    pzu .= (solver.xu_r .- solver.x_ur) .* solver.zu_r .- solver.mu

end
# function set_aug_rhs!(solver::MadNLPSolver, kkt::AbstractKKTSystem, c)
#     px = primal(solver.p)
#     x = primal(solver.x)
#     f = primal(solver.f)
#     xl = primal(solver.xl)
#     xu = primal(solver.xu)
#     zl = full(solver.zl)
#     zu = full(solver.zu)
#     @inbounds @simd for i in eachindex(px)
#         px[i] = -f[i] + zl[i] - zu[i] - solver.jacl[i]
#     end
    
#     py = dual(solver.p)
#     @inbounds @simd for i in eachindex(py)
#         py[i] = -c[i]
#     end

#     pzl = dual_lb(solver.p)
#     @inbounds @simd for i in eachindex(pzl)
#         pzl[i] = (solver.xl_r[i] - solver.x_lr[i]) * solver.zl_r[i] + solver.mu 
#     end

#     pzu = dual_ub(solver.p)
#     @inbounds @simd for i in eachindex(pzu)
#         pzu[i] = (solver.xu_r[i] -solver.x_ur[i]) * solver.zu_r[i] - solver.mu 
#     end
# return
# end 

# function set_aug_rhs!(solver::MadNLPSolver, kkt::SparseUnreducedKKTSystem, c)
#     f = primal(solver.f)
#     zl = primal(solver.zl)
#     zu = primal(solver.zu)
#     px = primal(solver.p)
#     @inbounds @simd for i in eachindex(px)
#         px[i] = -f[i] + zl[i] - zu[i] - solver.jacl[i]
#     end
#     py = dual(solver.p)
#     @inbounds @simd for i in eachindex(py)
#         py[i] = -c[i]
#     end
#     pzl = dual_lb(solver.p)
#     @inbounds @simd for i in eachindex(pzl)
#         pzl[i] = (solver.xl_r[i] - solver.x_lr[i]) * kkt.l_lower[i] + solver.mu / kkt.l_lower[i]
#     end
#     pzu = dual_ub(solver.p)
#     @inbounds @simd for i in eachindex(pzu)
#         pzu[i] = (solver.xu_r[i] - solver.x_ur[i]) * kkt.u_lower[i] - solver.mu / kkt.u_lower[i]
#     end
#     return
# end


# Set RHS RR
function set_aug_rhs_RR!(
    solver::MadNLPSolver, kkt::AbstractKKTSystem, RR::RobustRestorer, rho,
)
    x = full(solver.x)
    xl = full(solver.xl)
    xu = full(solver.xu)
    zl = full(solver.zl)
    zu = full(solver.zu)

    px = primal(solver.p)
    py = dual(solver.p)
    pzl = dual_lb(solver.p)
    pzu = dual_ub(solver.p)

    mu = RR.mu_R
    
    px .= .- RR.f_R .+ zl .- zu .- solver.jacl 
    py .= .- solver.c .+ RR.pp .- RR.nn .+
        (mu .- (rho .- solver.y) .* RR.pp) ./ RR.zp .-
        (mu .- (rho .+ solver.y) .* RR.nn) ./ RR.zn

    pzl .= (solver.xl_r .- solver.x_lr) .* solver.zl_r .+ mu
    pzu .= (solver.xu_r .- solver.x_ur) .* solver.zu_r .- mu

    return
end

# solving KKT system
@inbounds function _kktmul!(w,x,del_w,du_diag,l_lower,u_lower,l_diag,u_diag, alpha, beta)
    primal(w) .+= alpha .* del_w .* primal(x)
    dual(w) .+= alpha .* du_diag .* dual(x)
    w.xp_lr .-= alpha .* dual_lb(x)
    w.xp_ur .+= alpha .* dual_ub(x)
    dual_lb(w) .= beta .* dual_lb(w) .+ alpha .* (x.xp_lr .* l_lower .- dual_lb(x) .* l_diag)
    dual_ub(w) .= beta .* dual_ub(w) .+ alpha .* (x.xp_ur .* u_lower .+ dual_ub(x) .* u_diag)
end

@inbounds function reduce_rhs!(
    xp_lr,wl,l_diag,
    xp_ur,wu,u_diag,
    )
    xp_lr .-= wl ./ l_diag
    xp_ur .-= wu ./ u_diag
end


# Finish
# temporaily commented out
# function finish_aug_solve!(kkt::AbstractKKTSystem, d)
#     dlb = dual_lb(d)
#     dub = dual_ub(d)
#     @inbounds @simd for i in eachindex(dlb)
#         dlb[i] = (dlb[i] - kkt.l_lower[i] * d.xp_lr[i]) / kkt.l_diag[i]
#     end
#     @inbounds @simd for i in eachindex(dub)
#         dub[i] = (dub[i] - kkt.u_lower[i] * d.xp_ur[i]) / kkt.u_diag[i]
#     end

#     return
# end

function finish_aug_solve!(kkt::AbstractKKTSystem, d)
    dlb = dual_lb(d)
    dub = dual_ub(d)
    dlb .= (.-dlb .+ kkt.l_lower .* d.xp_lr) ./ kkt.l_diag
    dub .= (  dub .- kkt.u_lower .* d.xp_ur) ./ kkt.u_diag
    return
end
# function finish_aug_solve!(solver::MadNLPSolver, kkt::SparseUnreducedKKTSystem, mu)
#     dlb = dual_lb(solver.d)
#     @inbounds @simd for i in eachindex(dlb)
#         dlb[i] = (mu-solver.zl_r[i]*solver.dx_lr[i]) / (solver.x_lr[i]-solver.xl_r[i]) - solver.zl_r[i]
#     end
#     dub = dual_ub(solver.d)
#     @inbounds @simd for i in eachindex(dub)
#         dub[i] = (mu+solver.zu_r[i]*solver.dx_ur[i]) / (solver.xu_r[i]-solver.x_ur[i]) - solver.zu_r[i]
#     end
#     return
# end

# Initial
# temporaily commented out
# function set_initial_bounds!(solver::MadNLPSolver{T}) where T
#     @inbounds @simd for i in eachindex(solver.xl_r)
#         solver.xl_r[i] -= max(one(T),abs(solver.xl_r[i]))*solver.opt.tol
#     end
#     @inbounds @simd for i in eachindex(solver.xu_r)
#         solver.xu_r[i] += max(one(T),abs(solver.xu_r[i]))*solver.opt.tol
#     end
# end
function set_initial_bounds!(xl::AbstractVector{T},xu,tol) where T
    map!(
        x->x - max(one(T), abs(x)) .* tol,
        xl, xl
    )
    map!(
        x->x + max(one(T), abs(x)) .* tol,
        xu, xu
    )
end

function set_initial_rhs!(solver::MadNLPSolver{T}, kkt::AbstractKKTSystem) where T
    f = primal(solver.f)
    zl = primal(solver.zl)
    zu = primal(solver.zu)
    px = primal(solver.p)
    px .= .-f .+ zl .- zu
    fill!(dual(solver.p), zero(T))
    fill!(dual_lb(solver.p), zero(T))
    fill!(dual_ub(solver.p), zero(T))
    return
end

# Set ifr
function set_aug_rhs_ifr!(solver::MadNLPSolver{T}, kkt::AbstractKKTSystem) where T
    fill!(primal(solver._w1), zero(T))
    fill!(dual_lb(solver._w1), zero(T))
    fill!(dual_ub(solver._w1), zero(T))
    wy = dual(solver._w1)
    wy .= .- solver.c
    return
end

function set_g_ifr!(solver::MadNLPSolver, g)
    f = full(solver.f)
    x = full(solver.x)
    xl = full(solver.xl)
    xu = full(solver.xu)
    g .= f .- solver.mu ./ (x .- xl) .+ solver.mu ./ (xu .- x) .+ solver.jacl
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
is_valid(val::Real) = !(isnan(val) || isinf(val))
function is_valid(vec::AbstractArray)
    @inbounds @simd for i=1:length(vec)
        is_valid(vec[i]) || return false
    end
    return true
end
is_valid(args...) = all(is_valid(arg) for arg in args)

# temporaily commented out
function get_varphi(obj_val, x_lr::SubVector{T,Vector{T},VI}, xl_r, xu_r, x_ur, mu) where {T, VI}
    varphi = obj_val
    @inbounds @simd for i=1:length(x_lr)
        xll = x_lr[i]-xl_r[i]
        xll < 0 && return T(Inf)
        varphi -= mu*log(xll)
    end
    @inbounds @simd for i=1:length(x_ur)
        xuu = xu_r[i]-x_ur[i]
        xuu < 0 && return T(Inf)
        varphi -= mu*log(xuu)
    end
    return varphi
end
function get_varphi(obj_val, x_lr, xl_r, xu_r, x_ur, mu)
    
    return obj_val + mapreduce(
        (x1,x2) -> _get_varphi(x1,x2,mu), +, x_lr, xl_r
    ) + mapreduce(
        (x1,x2) -> _get_varphi(x1,x2,mu), +, xu_r, x_ur
    )
end

function _get_varphi(x1::T,x2,mu) where T
    x = x1 - x2
    if x < 0
        return T(Inf)
    else
        return -mu * log(x)
    end
end

@inline get_inf_pr(c) = norm(c, Inf)

# temporarily commented out
function get_inf_du(f::Vector{T}, zl, zu, jacl, sd) where T
    inf_du = 0.0
    @inbounds @simd for i=1:length(f)
        inf_du = max(inf_du,abs(f[i]-zl[i]+zu[i]+jacl[i]))
    end
    return inf_du/sd
end
function get_inf_du(f, zl, zu, jacl, sd)
    return mapreduce((f,zl,zu,jacl) -> abs(f-zl+zu+jacl), max, f, zl, zu, jacl; init = zero(eltype(f))) / sd
end

# temporarily commented out
function get_inf_compl(x_lr::SubVector{T,Vector{T},VI}, xl_r, zl_r, xu_r, x_ur, zu_r, mu, sc) where {T, VI}
    inf_compl = 0.0
    @inbounds @simd for i=1:length(x_lr)
        inf_compl = max(inf_compl,abs((x_lr[i]-xl_r[i])*zl_r[i]-mu))
    end
    @inbounds @simd for i=1:length(x_ur)
        inf_compl = max(inf_compl,abs((xu_r[i]-x_ur[i])*zu_r[i]-mu))
    end
    return inf_compl/sc
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

# temporarily commented out
function get_varphi_d(f::Vector{T}, x, xl, xu, dx, mu) where T
    varphi_d = 0.0
    @inbounds @simd for i=1:length(f)
        varphi_d += (f[i] - mu/(x[i]-xl[i]) + mu/(xu[i]-x[i])) * dx[i]
    end
    return varphi_d
end
function get_varphi_d(f, x, xl, xu, dx, mu)
    return mapreduce(
        (f,x,xl,xu,dx)-> (f - mu/(x-xl) + mu/(xu-x)) * dx,
        +,
        f, x, xl, xu, dx;
        init = zero(eltype(f))
    )
end

# temporarily commented out
function get_alpha_max(x::Vector{T}, xl, xu, dx, tau) where T
    alpha_max = 1.0
    @inbounds @simd for i=1:length(x)
        dx[i]<0 && (alpha_max=min(alpha_max,(-x[i]+xl[i])*tau/dx[i]))
        dx[i]>0 && (alpha_max=min(alpha_max,(-x[i]+xu[i])*tau/dx[i]))
    end
    return alpha_max
end
function get_alpha_max(x::VT, xl, xu, dx, tau) where {T, VT <: AbstractVector{T}}
    return min(
        mapreduce(
            (x, xl, dx) -> dx < 0 ? (-x+xl)*tau/dx : T(Inf),
            min,
            
            x, xl, dx,
            init = one(eltype(x))
        ),
        mapreduce(
            (x, xu, dx) -> dx > 0 ? (-x+xu)*tau/dx : T(Inf),
            min,
            x, xu, dx,
            init = one(eltype(x))
        )
    )
end


# temporarily commented out
function get_alpha_z(zl_r::SubVector{T,Vector{T},VI}, zu_r, dzl, dzu, tau) where {T, VI}
    alpha_z = 1.0
    @inbounds @simd for i=1:length(zl_r)
        dzl[i] < 0 && (alpha_z=min(alpha_z,-zl_r[i]*tau/dzl[i]))
     end
    @inbounds @simd for i=1:length(zu_r)
        dzu[i] < 0 && (alpha_z=min(alpha_z,-zu_r[i]*tau/dzu[i]))
    end
    return alpha_z
end
function get_alpha_z(zl_r::VT, zu_r, dzl, dzu, tau)  where {T, VT <: AbstractVector{T}}
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

function get_obj_val_R(p::Vector{T}, n, D_R, x, x_ref, rho, zeta) where T
    obj_val_R = 0.
    @inbounds @simd for i=1:length(p)
        obj_val_R += rho*(p[i]+n[i]) .+ zeta/2*D_R[i]^2*(x[i]-x_ref[i])^2
    end
    return obj_val_R
end
function get_obj_val_R(p::VT, n, D_R, x, x_ref, rho, zeta) where {T, VT <: AbstractVector{T}}
    return mapreduce(
        (p,n,D_R,x,x_ref) -> rho*(p+n) .+ zeta/2*D_R^2*(x-x_ref)^2,
        +,
        p,n,D_R,x,x_ref;
        init = zero(T)
    )
end

@inline get_theta(c) = norm(c, 1)

function get_theta_R(c::Vector{T}, p, n) where T
    theta_R = 0.0
    @inbounds @simd for i=1:length(c)
        theta_R += abs(c[i]-p[i]+n[i])
    end
    return theta_R
end
function get_theta_R(c::VT, p, n) where {T, VT <: AbstractVector{T}}
    return mapreduce(
        (c,p,n) -> abs(c-p+n),
        +,
        c,p,n;
        init = zero(T)
    )
end

function get_inf_pr_R(c::Vector{T}, p, n) where T
    inf_pr_R = 0.0
    @inbounds @simd for i=1:length(c)
        inf_pr_R = max(inf_pr_R,abs(c[i]-p[i]+n[i]))
    end
    return inf_pr_R
end
function get_inf_pr_R(c::VT, p, n) where {T, VT <: AbstractVector{T}}
    return mapreduce(
        (c,p,n) -> abs(c-p+n),
        max,
        c,p,n;
        init = zero(T)
    )
end

function get_inf_du_R(f_R::Vector{T}, l, zl, zu, jacl, zp, zn, rho, sd) where T
    inf_du_R = 0.0
    @inbounds @simd for i=1:length(zl)
        inf_du_R = max(inf_du_R,abs(f_R[i]-zl[i]+zu[i]+jacl[i]))
    end
    @inbounds @simd for i=1:length(zp)
        inf_du_R = max(inf_du_R,abs(rho-l[i]-zp[i]))
    end
    @inbounds @simd for i=1:length(zn)
        inf_du_R = max(inf_du_R,abs(rho+l[i]-zn[i]))
    end
    return inf_du_R / sd
end
function get_inf_du_R(f_R::VT, l, zl, zu, jacl, zp, zn, rho, sd)  where {T, VT <: AbstractVector{T}}
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
    x_lr::SubVector{T,Vector{T},VI}, xl_r, zl_r, xu_r, x_ur, zu_r, pp, zp, nn, zn, mu_R, sc
    ) where {T, VI}
    
    inf_compl_R = 0.0
    @inbounds @simd for i=1:length(x_lr)
        inf_compl_R = max(inf_compl_R,abs((x_lr[i]-xl_r[i])*zl_r[i]-mu_R))
    end
    @inbounds @simd for i=1:length(xu_r)
        inf_compl_R = max(inf_compl_R,abs((xu_r[i]-x_ur[i])*zu_r[i]-mu_R))
    end
    @inbounds @simd for i=1:length(pp)
        inf_compl_R = max(inf_compl_R,abs(pp[i]*zp[i]-mu_R))
    end
    @inbounds @simd for i=1:length(nn)
        inf_compl_R = max(inf_compl_R,abs(nn[i]*zn[i]-mu_R))
    end
    return inf_compl_R / sc
end
function get_inf_compl_R(
    x_lr::SubVector{T,VT,VI}, xl_r, zl_r, xu_r, x_ur, zu_r, pp, zp, nn, zn, mu_R, sc
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
    )/ sc
end

function get_alpha_max_R(x::Vector{T}, xl, xu, dx, pp, dpp, nn, dnn, tau_R) where T
    alpha_max_R = 1.0
    @inbounds @simd for i=1:length(x)
        dx[i]<0 && (alpha_max_R=min(alpha_max_R,(-x[i]+xl[i])*tau_R/dx[i]))
        dx[i]>0 && (alpha_max_R=min(alpha_max_R,(-x[i]+xu[i])*tau_R/dx[i]))
    end
    @inbounds @simd for i=1:length(pp)
        dpp[i]<0 && (alpha_max_R=min(alpha_max_R,-pp[i]*tau_R/dpp[i]))
    end
    @inbounds @simd for i=1:length(nn)
        dnn[i]<0 && (alpha_max_R=min(alpha_max_R,-nn[i]*tau_R/dnn[i]))
    end
    return alpha_max_R
end
function get_alpha_max_R(x::VT, xl, xu, dx, pp, dpp, nn, dnn, tau_R) where {T, VT <: AbstractVector{T}}
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
    zl_r::SubVector{T,Vector{T},VI}, zu_r, dzl, dzu, zp, dzp, zn, dzn, tau_R
    ) where {T, VI}
    
    alpha_z_R = 1.0
    @inbounds @simd for i=1:length(zl_r)
        dzl[i]<0 && (alpha_z_R=min(alpha_z_R,-zl_r[i]*tau_R/dzl[i]))
    end
    @inbounds @simd for i=1:length(zu_r)
        dzu[i]<0 && (alpha_z_R=min(alpha_z_R,-zu_r[i]*tau_R/dzu[i]))
    end
    @inbounds @simd for i=1:length(zp)
        dzp[i]<0 && (alpha_z_R=min(alpha_z_R,-zp[i]*tau_R/dzp[i]))
    end
    @inbounds @simd for i=1:length(zn)
        dzn[i]<0 && (alpha_z_R=min(alpha_z_R,-zn[i]*tau_R/dzn[i]))
    end
    return alpha_z_R
end
function get_alpha_z_R(
    zl_r::SubVector{T,VT,VI}, zu_r, dzl, dzu, zp, dzp, zn, dzn, tau_R
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
    obj_val, x_lr::SubVector{T,Vector{T},VI}, xl_r, xu_r, x_ur, pp, nn, mu_R
    )  where {T, VI}
    
    varphi_R = obj_val
    @inbounds @simd for i=1:length(x_lr)
        xll = x_lr[i]-xl_r[i]
        xll < 0 && return T(Inf)
        varphi_R -= mu_R*log(xll)
    end
    @inbounds @simd for i=1:length(x_ur)
        xuu = xu_r[i]-x_ur[i]
        xuu < 0 && return T(Inf)
        varphi_R -= mu_R*log(xuu)
    end
    @inbounds @simd for i=1:length(pp)
        pp[i] < 0 && return T(Inf)
        varphi_R -= mu_R*log(pp[i])
    end
    @inbounds @simd for i=1:length(pp)
        nn[i] < 0 && return T(Inf)
        varphi_R -= mu_R*log(nn[i])
    end
    return varphi_R
end
function get_varphi_R(
    obj_val, x_lr::SubVector{T,VT,VI}, xl_r, xu_r, x_ur, pp, nn, mu_R
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
            x_lr, xl_r;
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


function get_F(c::Vector{T}, f, zl, zu, jacl, x_lr, xl_r, zl_r, xu_r, x_ur, zu_r, mu) where T
    F = 0.0
    @inbounds @simd for i=1:length(c)
        F += abs(c[i])
    end
    @inbounds @simd for i=1:length(f)
        F += abs(f[i]-zl[i]+zu[i]+jacl[i])
    end
    @inbounds @simd for i=1:length(x_lr)
        x_lr[i] >= xl_r[i] || return T(Inf)
        zl_r[i] >= 0       || return T(Inf)
        F += abs((x_lr[i]-xl_r[i])*zl_r[i]-mu)
    end
    @inbounds @simd for i=1:length(x_ur)
        xu_r[i] >= x_ur[i] || return T(Inf)
        zu_r[i] >= 0       || return T(Inf)
        F += abs((xu_r[i]-xu_r[i])*zu_r[i]-mu)
    end
    return F
end
function get_F(c::AbstractVector{T}, f, zl, zu, jacl, x_lr, xl_r, zl_r, xu_r, x_ur, zu_r, mu) where T
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


function get_varphi_d_R(f_R::Vector{T}, x, xl, xu, dx, pp, nn, dpp, dnn, mu_R, rho) where T
    varphi_d = 0.0
    @inbounds @simd for i=1:length(x)
        varphi_d += (f_R[i] - mu_R/(x[i]-xl[i]) + mu_R/(xu[i]-x[i])) * dx[i]
    end
    @inbounds @simd for i=1:length(pp)
        varphi_d += (rho - mu_R/pp[i]) * dpp[i]
    end
    @inbounds @simd for i=1:length(nn)
        varphi_d += (rho - mu_R/nn[i]) * dnn[i]
    end
    return varphi_d
end
function get_varphi_d_R(f_R::VT, x, xl, xu, dx, pp, nn, dpp, dnn, mu_R, rho) where {T, VT <: AbstractVector{T}}
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

# temporarily commented out
# function initialize_variables!(x, xl, xu, bound_push, bound_fac)
#     @inbounds @simd for i=1:length(x)
#         if xl[i]!=-Inf && xu[i]!=Inf
#             x[i] = min(
#                 xu[i]-min(bound_push*max(1,abs(xu[i])), bound_fac*(xu[i]-xl[i])),
#                 max(xl[i]+min(bound_push*max(1,abs(xl[i])),bound_fac*(xu[i]-xl[i])),x[i]),
#             )
#         elseif xl[i]!=-Inf && xu[i]==Inf
#             x[i] = max(xl[i]+bound_push*max(1,abs(xl[i])), x[i])
#         elseif xl[i]==-Inf && xu[i]!=Inf
#             x[i] = min(xu[i]-bound_push*max(1,abs(xu[i])), x[i])
#         end
#     end
# end

function initialize_variables!(x, xl, xu, bound_push, bound_fac)
    map!((x,l,u) -> _initialize_variables!(x,l,u, bound_push, bound_fac), x, x, xl, xu)
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



function adjust_boundary!(x_lr::VT, xl_r, x_ur, xu_r, mu) where {T, VT <: AbstractVector{T}}
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

# temporarily commented out
function get_rel_search_norm(x::Vector{T}, dx) where T
    rel_search_norm = 0.0
    @inbounds @simd for i=1:length(x)
        rel_search_norm = max(
            rel_search_norm,
            abs(dx[i]) / (1.0 + abs(x[i])),
        )
    end
    return rel_search_norm
end
function get_rel_search_norm(x, dx)
    return mapreduce(
        (x,dx) -> abs(dx) / (1.0 + abs(x)),
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

function get_mu(mu, mu_min, mu_linear_decrease_factor, mu_superlinear_decrease_power, tol)
    # Warning: `a * tol` should be strictly less than 100 * mu_min, see issue #242
    a = min(99.0 * mu_min / tol, 0.01)
    return max(
        mu_min,
        a * tol,
        min(mu_linear_decrease_factor*mu, mu^mu_superlinear_decrease_power),
    )
end

@inline get_tau(mu, tau_min) = max(tau_min, 1-mu)

function get_alpha_min(theta, varphi_d, theta_min, gamma_theta, gamma_phi, alpha_min_frac, del, s_theta, s_phi)
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

function is_barr_obj_rapid_increase(varphi, varphi_trial, obj_max_inc)
    return (varphi_trial >= varphi) && (log10(varphi_trial-varphi) > obj_max_inc + max(1.0, log10(abs(varphi))))
end

# temporarily commented out
# function reset_bound_dual!(z, x, mu, kappa_sigma)
#     @inbounds @simd for i in eachindex(z)
#         z[i] = max(min(z[i], (kappa_sigma*mu)/x[i]), (mu/kappa_sigma)/x[i])
#     end
#     return
# end
function reset_bound_dual!(z, x, mu, kappa_sigma)
    map!(
        (z, x) -> max(min(z, (kappa_sigma*mu)/x), (mu/kappa_sigma)/x),
        z, z, x
    )
    return
end

# temporarily commented out
# function reset_bound_dual!(z, x1, x2, mu, kappa_sigma)
#     @inbounds @simd for i in eachindex(z)
#         z[i] = max(min(z[i], (kappa_sigma*mu)/(x1[i]-x2[i])), (mu/kappa_sigma)/(x1[i]-x2[i]))
#     end
#     return
# end

function reset_bound_dual!(z, x1, x2, mu, kappa_sigma)
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

