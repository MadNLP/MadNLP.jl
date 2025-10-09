function RobustRestorer(solver::AbstractMadNLPSolver{T}) where {T}

    f_R = similar(_y(solver), _n(solver))
    x_ref = similar(_y(solver), _n(solver))
    D_R = similar(_y(solver), _n(solver))
    pp = similar(_y(solver), _m(solver))
    nn = similar(_y(solver), _m(solver))
    pp_trial = similar(_y(solver), _m(solver))
    nn_trial = similar(_y(solver), _m(solver))

    nn = similar(_y(solver), _m(solver))
    zp = similar(_y(solver), _m(solver))
    zn = similar(_y(solver), _m(solver))
    dpp= similar(_y(solver), _m(solver))
    dnn= similar(_y(solver), _m(solver))
    dzp= similar(_y(solver), _m(solver))
    dzn= similar(_y(solver), _m(solver))
    pp_trial = similar(_y(solver), _m(solver))
    nn_trial = similar(_y(solver), _m(solver))

    return RobustRestorer(
        zero(T), 
        f_R, 
        x_ref, 
        zero(T), 
        D_R, 
        zero(T), 
        pp, 
        nn, 
        zp, zn, 
        dpp, dnn, dzp, dzn, 
        pp_trial, 
        nn_trial, 
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), 
        Tuple{T, T}[], 
    )
end

function initialize_robust_restorer!(solver::AbstractMadNLPSolver{T}) where T
    @trace(_logger(solver),"Initializing restoration phase variables.")
    _RR(solver) == nothing && (_RR!(solver, RobustRestorer(solver)))
    RR = _RR(solver)

    copyto!(RR.x_ref, full(_x(solver)))
    RR.theta_ref = get_theta(_c(solver))
    RR.D_R .= min.(one(T), one(T) ./ abs.(RR.x_ref))

    RR.mu_R = max(_mu(solver), norm(_c(solver), Inf))
    RR.tau_R= max(_opt(solver).tau_min,1-RR.mu_R)
    RR.zeta = sqrt(RR.mu_R)

    rho = _opt(solver).rho
    mu = RR.mu_R
    RR.nn .=
        (mu .- rho*_c(solver))./2 ./rho .+
        sqrt.(
            ((mu.-rho*_c(solver))./2 ./rho).^2 + mu.*_c(solver)./2 ./rho
        )
    RR.pp .= _c(solver) .+ RR.nn
    RR.zp .= RR.mu_R ./ RR.pp
    RR.zn .= RR.mu_R ./ RR.nn

    RR.obj_val_R = get_obj_val_R(RR.pp,RR.nn,RR.D_R,full(_x(solver)),RR.x_ref,_opt(solver).rho,RR.zeta)
    fill!(RR.f_R, zero(T))
    empty!(RR.filter)
    push!(RR.filter, (_theta_max(solver),-Inf))

    fill!(_y(solver), zero(T))
    _zl_r(solver) .= min.(_opt(solver).rho, _zl_r(solver))
    _zu_r(solver) .= min.(_opt(solver).rho, _zu_r(solver))
    # fill!(_zl_r(solver), one(T)) # Experimental
    # fill!(_zu_r(solver), one(T)) # Experimental
    
    _cnt(solver).t = 0

    # misc
    _del_w!(solver, zero(T))
end

