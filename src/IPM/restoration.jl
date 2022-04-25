mutable struct RobustRestorer
    obj_val_R::Float64
    f_R::Vector{Float64}
    x_ref::Vector{Float64}

    theta_ref::Float64
    D_R::Vector{Float64}
    obj_val_R_trial::Float64

    pp::Vector{Float64}
    nn::Vector{Float64}
    zp::Vector{Float64}
    zn::Vector{Float64}

    dpp::Vector{Float64}
    dnn::Vector{Float64}
    dzp::Vector{Float64}
    dzn::Vector{Float64}

    pp_trial::Vector{Float64}
    nn_trial::Vector{Float64}

    inf_pr_R::Float64
    inf_du_R::Float64
    inf_compl_R::Float64

    mu_R::Float64
    tau_R::Float64
    zeta::Float64

    filter::Vector{Tuple{Float64,Float64}}
end

function RobustRestorer(ips::AbstractInteriorPointSolver)

    nn = Vector{Float64}(undef,ips.m)
    zp = Vector{Float64}(undef,ips.m)
    zn = Vector{Float64}(undef,ips.m)
    dpp= Vector{Float64}(undef,ips.m)
    dnn= Vector{Float64}(undef,ips.m)
    dzp= Vector{Float64}(undef,ips.m)
    dzn= Vector{Float64}(undef,ips.m)
    pp_trial = Vector{Float64}(undef,ips.m)
    nn_trial = Vector{Float64}(undef,ips.m)

    return RobustRestorer(0.,ips._w2x,ips._w1x,0.,ips._w3x,0.,ips._w3l,ips._w4l,
                          zp,zn,dpp,dnn,dzp,dzn,ips._w2l,ips._w1l,
                          0.,0.,0.,0.,0.,0.,Tuple{Float64,Float64}[])
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

