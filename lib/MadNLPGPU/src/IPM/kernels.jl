# TODO(@anton): All of these could probably be made more efficient via custom kernels.
#               For now I am just returning the mapreduce functionality here so MadNLPGPU works again.

const AbstractGPUVectorOrSubVector{T,VT<:AbstractGPUVector{T}} = Union{AbstractGPUVector{T}, SubVector{T, VT}}

function get_varphi(
    obj_val,
    x_lr::VT,
    xl_r::VT,
    xu_r::VT,
    x_ur::VT,
    mu
) where {T, VT <: AbstractGPUVectorOrSubVector{T}}
    return obj_val + mapreduce(
        (x1,x2) -> _get_varphi(x1,x2,mu), +, x_lr, xl_r
    ) + mapreduce(
        (x1,x2) -> _get_varphi(x1,x2,mu), +, xu_r, x_ur
    )
end

function get_inf_du(f::VT, zl::VT, zu::VT, jacl::VT, sd) where {VT <: AbstractGPUVectorOrSubVector}
    return mapreduce((f,zl,zu,jacl) -> abs(f-zl+zu+jacl), max, f, zl, zu, jacl; init = zero(eltype(f))) / sd
end


function get_inf_compl(x_lr::VT, xl_r::VT, zl_r::VT, xu_r::VT, x_ur::VT, zu_r::VT, mu, sc) where {VT <: AbstractGPUVectorOrSubVector}
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

function get_min_complementarity(
    x_lr::AbstractGPUVectorOrSubVector{T},
    xl_r::AbstractGPUVectorOrSubVector{T},
    zl_r::AbstractGPUVectorOrSubVector{T},
    x_ur::AbstractGPUVectorOrSubVector{T},
    xu_r::AbstractGPUVectorOrSubVector{T},
    zu_r::AbstractGPUVectorOrSubVector{T}
) where T
    cc_lb = mapreduce((x_l, xl, zl) -> (x_l-xl)*zl, min, x_lr, xl_r, zl_r, init=T(Inf))
    cc_ub = mapreduce((x_u, xu, zu) -> (xu-x_u)*zu, min, x_ur, xu_r, zu_r, init=T(Inf))
    return min(cc_lb,cc_ub)
end

function get_varphi_d(
    f::AbstractGPUVectorOrSubVector{T},
    x::AbstractGPUVectorOrSubVector{T},
    xl::AbstractGPUVectorOrSubVector{T},
    xu::AbstractGPUVectorOrSubVector{T},
    dx::AbstractGPUVectorOrSubVector{T},
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
    x::AbstractGPUVectorOrSubVector{T},
    xl::AbstractGPUVectorOrSubVector{T},
    xu::AbstractGPUVectorOrSubVector{T},
    dx::AbstractGPUVectorOrSubVector{T},
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
    zl_r::AbstractGPUVectorOrSubVector{T},
    zu_r::AbstractGPUVectorOrSubVector{T},
    dzl::AbstractGPUVectorOrSubVector{T},
    dzu::AbstractGPUVectorOrSubVector{T},
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
    p::AbstractGPUVectorOrSubVector{T},
    n::AbstractGPUVectorOrSubVector{T},
    D_R::AbstractGPUVectorOrSubVector{T},
    x::AbstractGPUVectorOrSubVector{T},
    x_ref::AbstractGPUVectorOrSubVector{T},
    rho,
    zeta,
) where T
    return mapreduce(
        (p,n) -> rho*(p+n),
        +,
        p,n;
        init = zero(T)
    ) +
        mapreduce(
            (D_R,x,x_ref) -> zeta/2*D_R^2*(x-x_ref)^2,
            +,
            D_R,x,x_ref;
            init = zero(T)
        )
end

function get_theta_R(
    c::AbstractGPUVectorOrSubVector{T},
    p::AbstractGPUVectorOrSubVector{T},
    n::AbstractGPUVectorOrSubVector{T},
) where T
    return mapreduce(
        (c,p,n) -> abs(c-p+n),
        +,
        c,p,n;
        init = zero(T)
    )
end

function get_inf_pr_R(
    c::AbstractGPUVectorOrSubVector{T},
    p::AbstractGPUVectorOrSubVector{T},
    n::AbstractGPUVectorOrSubVector{T},
) where T
    return mapreduce(
        (c,p,n) -> abs(c-p+n),
        max,
        c,p,n;
        init = zero(T)
    )
end

function get_inf_du_R(
    f_R::AbstractGPUVectorOrSubVector{T},
    l::AbstractGPUVectorOrSubVector{T},
    zl::AbstractGPUVectorOrSubVector{T},
    zu::AbstractGPUVectorOrSubVector{T},
    jacl::AbstractGPUVectorOrSubVector{T},
    zp::AbstractGPUVectorOrSubVector{T},
    zn::AbstractGPUVectorOrSubVector{T},
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
    x_lr::AbstractGPUVectorOrSubVector{T},
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
) where {T}
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
    x::AbstractGPUVectorOrSubVector{T},
    xl::AbstractGPUVectorOrSubVector{T},
    xu::AbstractGPUVectorOrSubVector{T},
    dx::AbstractGPUVectorOrSubVector{T},
    pp::AbstractGPUVectorOrSubVector{T},
    dpp::AbstractGPUVectorOrSubVector{T},
    nn::AbstractGPUVectorOrSubVector{T},
    dnn::AbstractGPUVectorOrSubVector{T},
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
    zl_r::AbstractGPUVectorOrSubVector{T},
    zu_r,
    dzl,
    dzu,
    zp,
    dzp,
    zn,
    dzn,
    tau_R,
) where {T}

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
    x_lr::AbstractGPUVectorOrSubVector{T},
    xl_r,
    xu_r,
    x_ur,
    pp,
    nn,
    mu_R,
)  where {T}
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
    c::AbstractGPUVectorOrSubVector{T},
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
    f_R::AbstractGPUVectorOrSubVector{T},
    x::AbstractGPUVectorOrSubVector{T},
    xl::AbstractGPUVectorOrSubVector{T},
    xu::AbstractGPUVectorOrSubVector{T},
    dx::AbstractGPUVectorOrSubVector{T},
    pp::AbstractGPUVectorOrSubVector{T},
    nn::AbstractGPUVectorOrSubVector{T},
    dpp::AbstractGPUVectorOrSubVector{T},
    dnn::AbstractGPUVectorOrSubVector{T},
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

function get_rel_search_norm(x::AbstractGPUVectorOrSubVector{T}, dx::AbstractGPUVectorOrSubVector{T}) where T
    return mapreduce(
        (x,dx) -> abs(dx) / (one(T) + abs(x)),
        max,
        x, dx
    )
end

function populate_RR_nn!(nn::AbstractGPUVectorOrSubVector{T}, c::AbstractGPUVectorOrSubVector{T}, mu, rho) where T
    map!((c) -> (mu - rho*c)/(2*rho)+sqrt(((mu-rho*c)/(2*rho))^2 + mu*c/(2*rho)), nn, c)
end
