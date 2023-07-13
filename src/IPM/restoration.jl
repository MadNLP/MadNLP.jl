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

function RobustRestorer(solver::AbstractMadNLPSolver{T}) where {T}

    nn = similar(solver.y, solver.m)
    zp = similar(solver.y, solver.m)
    zn = similar(solver.y, solver.m)
    dpp= similar(solver.y, solver.m)
    dnn= similar(solver.y, solver.m)
    dzp= similar(solver.y, solver.m)
    dzn= similar(solver.y, solver.m)
    pp_trial = similar(solver.y, solver.m)
    nn_trial = similar(solver.y, solver.m)

    return RobustRestorer(
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

    copyto!(RR.x_ref, full(solver.x))
    RR.theta_ref = get_theta(solver.c)
    RR.D_R .= min.(one(T), one(T) ./ abs.(RR.x_ref))

    RR.mu_R = max(solver.mu, norm(solver.c, Inf))
    RR.tau_R= max(solver.opt.tau_min,1-RR.mu_R)
    RR.zeta = sqrt(RR.mu_R)

    rho = solver.opt.rho
    mu = RR.mu_R
    RR.nn .=
        (mu .- rho*solver.c)./2 ./rho .+
        sqrt.(
            ((mu.-rho*solver.c)./2 ./rho).^2 + mu.*solver.c./2 ./rho
        )
    RR.pp .= solver.c .+ RR.nn
    RR.zp .= RR.mu_R ./ RR.pp
    RR.zn .= RR.mu_R ./ RR.nn

    RR.obj_val_R = get_obj_val_R(RR.pp,RR.nn,RR.D_R,full(solver.x),RR.x_ref,solver.opt.rho,RR.zeta)
    fill!(RR.f_R, zero(T))
    empty!(RR.filter)
    push!(RR.filter, (solver.theta_max,-Inf))

    fill!(solver.y, zero(T))
    solver.zl_r .= min.(solver.opt.rho, solver.zl_r)
    solver.zu_r .= min.(solver.opt.rho, solver.zu_r)
    
    solver.cnt.t = 0

    # misc
    solver.kkt.del_w = zero(T)
end

