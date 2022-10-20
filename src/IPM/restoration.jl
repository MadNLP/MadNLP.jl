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

function RobustRestorer(solver::AbstractMadNLPSolver{T}) where T

    nn = Vector{T}(undef,solver.m)
    zp = Vector{T}(undef,solver.m)
    zn = Vector{T}(undef,solver.m)
    dpp= Vector{T}(undef,solver.m)
    dnn= Vector{T}(undef,solver.m)
    dzp= Vector{T}(undef,solver.m)
    dzn= Vector{T}(undef,solver.m)
    pp_trial = Vector{T}(undef,solver.m)
    nn_trial = Vector{T}(undef,solver.m)

    return RobustRestorer{T}(
        0.,
        primal(solver._w2),
        primal(solver._w1),
        0.,
        primal(solver._w3),
        0.,
        dual(solver._w3),
        dual(solver._w4),
        zp, zn,
        dpp, dnn, dzp, dzn,
        dual(solver._w2),
        dual(solver._w1),
        0.,0.,0.,0.,0.,0.,
        Tuple{T,T}[],
    )
end

function initialize_robust_restorer!(solver::AbstractMadNLPSolver{T}) where T
    @trace(solver.logger,"Initializing restoration phase variables.")
    solver.RR == nothing && (solver.RR = RobustRestorer(solver))
    RR = solver.RR

    copyto!(RR.x_ref, solver.x)
    RR.theta_ref = get_theta(solver.c)
    RR.D_R .= min.(one(T), one(T) ./abs.(RR.x_ref))

    RR.mu_R = max(solver.mu, norm(solver.c, Inf))
    RR.tau_R= max(solver.opt.tau_min,1-RR.mu_R)
    RR.zeta = sqrt(RR.mu_R)

    RR.nn .= (RR.mu_R.-solver.opt.rho.*solver.c)./2 ./solver.opt.rho .+
        sqrt.(((RR.mu_R.-solver.opt.rho.*solver.c)./2 ./solver.opt.rho).^2 .+ RR.mu_R.*solver.c./2 ./solver.opt.rho)
    RR.pp .= solver.c .+ RR.nn
    RR.zp .= RR.mu_R./RR.pp
    RR.zn .= RR.mu_R./RR.nn

    RR.obj_val_R = get_obj_val_R(RR.pp,RR.nn,RR.D_R,solver.x,RR.x_ref,solver.opt.rho,RR.zeta)
    fill!(RR.f_R, zero(T))
    empty!(RR.filter)
    push!(RR.filter, (solver.theta_max,-Inf))

    fill!(solver.y, zero(T))
    solver.zl_r .= min.(solver.opt.rho, solver.zl_r)
    solver.zu_r .= min.(solver.opt.rho, solver.zu_r)
    solver.cnt.t = 0

    # misc
    solver.del_w = zero(T)
end

