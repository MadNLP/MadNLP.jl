mutable struct RobustRestorer{T} 
    obj_val_R::T
    f_R::Vector{T}
    x_ref::Vector{T}

    theta_ref::T
    D_R::Vector{T}
    obj_val_R_trial::T

    pp::Vector{T}
    nn::Vector{T}
    zp::Vector{T}
    zn::Vector{T}

    dpp::Vector{T}
    dnn::Vector{T}
    dzp::Vector{T}
    dzn::Vector{T}

    pp_trial::Vector{T}
    nn_trial::Vector{T}

    inf_pr_R::T
    inf_du_R::T
    inf_compl_R::T

    mu_R::T
    tau_R::T
    zeta::T

    filter::Vector{Tuple{T,T}}
end

function RobustRestorer(ips::AbstractInteriorPointSolver{T}) where T

    nn = Vector{T}(undef,ips.m)
    zp = Vector{T}(undef,ips.m)
    zn = Vector{T}(undef,ips.m)
    dpp= Vector{T}(undef,ips.m)
    dnn= Vector{T}(undef,ips.m)
    dzp= Vector{T}(undef,ips.m)
    dzn= Vector{T}(undef,ips.m)
    pp_trial = Vector{T}(undef,ips.m)
    nn_trial = Vector{T}(undef,ips.m)

    return RobustRestorer{T}(0.,ips._w2x,ips._w1x,0.,ips._w3x,0.,ips._w3l,ips._w4l,
                             zp,zn,dpp,dnn,dzp,dzn,ips._w2l,ips._w1l,
                             0.,0.,0.,0.,0.,0.,Tuple{T,T}[])
end

function initialize_robust_restorer!(ips::AbstractInteriorPointSolver)
    @trace(ips.logger,"Initializing restoration phase variables.")
    ips.RR == nothing && (ips.RR = RobustRestorer(ips))
    RR = ips.RR

    RR.x_ref .= ips.x
    RR.theta_ref = get_theta(ips.c)
    RR.D_R   .= min.(1,1 ./abs.(RR.x_ref))

    RR.mu_R = max(ips.mu,norm(ips.c,Inf))
    RR.tau_R= max(ips.opt.tau_min,1-RR.mu_R)
    RR.zeta = sqrt(RR.mu_R)

    RR.nn .= (RR.mu_R.-ips.opt.rho.*ips.c)./2 ./ips.opt.rho .+
        sqrt.(((RR.mu_R.-ips.opt.rho.*ips.c)./2 ./ips.opt.rho).^2 .+ RR.mu_R.*ips.c./2 ./ips.opt.rho)
    RR.pp .= ips.c .+ RR.nn
    RR.zp .= RR.mu_R./RR.pp
    RR.zn .= RR.mu_R./RR.nn

    RR.obj_val_R = get_obj_val_R(RR.pp,RR.nn,RR.D_R,ips.x,RR.x_ref,ips.opt.rho,RR.zeta)
    RR.f_R.=0
    empty!(RR.filter)
    push!(RR.filter,(ips.theta_max,-Inf))

    ips.l .= 0.
    ips.zl_r .= min.(ips.opt.rho,ips.zl_r)
    ips.zu_r .= min.(ips.opt.rho,ips.zu_r)
    ips.cnt.t = 0

    # misc
    ips.del_w = 0
end

