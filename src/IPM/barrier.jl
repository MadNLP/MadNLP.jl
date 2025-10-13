
"""
    update_barrier!(barrier::AbstractBarrierUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T

Update barrier using the rule in `barrier`. Store the results in `solver.mu`.

"""
function update_barrier! end

# Implement monotone update in a dedicated function as this code is reused
# throughout all the barrier updates.
function _update_monotone!(
    barrier::AbstractBarrierUpdate{T},
    solver::AbstractMadNLPSolver{T},
    sc::T
) where T
    inf_compl_mu = get_inf_compl(
        _x_lr(solver),
        _xl_r(solver),
        _zl_r(solver),
        _xu_r(solver),
        _x_ur(solver),
        _zu_r(solver),
        _mu(solver),
        sc,
    )
    while (_mu(solver) > max(barrier.mu_min, _opt(solver).tol/10)) &&
        (max(_inf_pr(solver), _inf_du(solver), inf_compl_mu) <= _opt(solver).barrier_tol_factor*_mu(solver))
        mu_new = get_mu(
            _mu(solver),
            barrier.mu_min,
            barrier.mu_linear_decrease_factor,
            barrier.mu_superlinear_decrease_power,
            _opt(solver).tol,
        )
        inf_compl_mu = get_inf_compl(
            _x_lr(solver),
            _xl_r(solver),
            _zl_r(solver),
            _xu_r(solver),
            _x_ur(solver),
            _zu_r(solver),
            _mu(solver),
            sc,
        )
        set_tau!(solver, get_tau(_mu(solver), _opt(solver).tau_min))
        set_mu!(solver, mu_new)
        empty!(_filter(solver))
        push!(_filter(solver), (_theta_max(solver), -Inf))
    end
    return
end

# Implement monotone update for feasibility restoration.
# N.B: we always fallack to monotone update for restoration, no matter
# the barrier update being used in the regular phase.
function _update_monotone_RR!(
    barrier::AbstractBarrierUpdate{T},
    solver::AbstractMadNLPSolver{T},
    sc::T
) where T
    RR = _RR(solver)
    inf_compl_mu_R = get_inf_compl_R(
        _x_lr(solver),
        _xl_r(solver),
        _zl_r(solver),
        _xu_r(solver),
        _x_ur(solver),
        _zu_r(solver),
        RR.pp,
        RR.zp,
        RR.nn,
        RR.zn,
        RR.mu_R,
        sc,
    )
    while RR.mu_R >= barrier.mu_min &&
        max(RR.inf_pr_R,RR.inf_du_R,inf_compl_mu_R) <= _opt(solver).barrier_tol_factor*RR.mu_R
        RR.mu_R = get_mu(
            RR.mu_R,
            barrier.mu_min,
            barrier.mu_linear_decrease_factor,
            barrier.mu_superlinear_decrease_power,
            _opt(solver).tol,
        )
        inf_compl_mu_R = get_inf_compl_R(
            _x_lr(solver),
            _xl_r(solver),
            _zl_r(solver),
            _xu_r(solver),
            _x_ur(solver),
            _zu_r(solver),
            RR.pp,
            RR.zp,
            RR.nn,
            RR.zn,
            RR.mu_R,
            sc,
        )
        RR.tau_R= max(_opt(solver).tau_min,1-RR.mu_R)
        RR.zeta = sqrt(RR.mu_R)
        empty!(RR.filter)
        push!(RR.filter,(_theta_max(solver),-Inf))
    end
    return
end

function MonotoneUpdate(tol::T, barrier_tol_factor) where T
    return MonotoneUpdate{T}(; mu_min=min(1e-4, tol ) / (barrier_tol_factor + 1))
end

function update_barrier!(barrier::MonotoneUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T
    _update_monotone!(barrier, solver, sc)
end

# Initial barrier parameter used when we fallback to the monotone barrier update.
function get_fixed_mu(solver::AbstractMadNLPSolver{T}, barrier::AbstractAdaptiveUpdate{T}) where T
    mu = T(0.8) * get_average_complementarity(solver)
    return clamp(mu, barrier.mu_min, barrier.mu_max)
end

function _check_progress(barrier::AbstractAdaptiveUpdate{T}, solver::AbstractMadNLPSolver{T}) where T
    if !barrier.globalization
        return true
    end
    kappa_1 = T(1e-5) # filter margin width
    kappa_2 = T(1.0)  # filter margin maximum width
    # Check current progress using filter line search
    theta = get_theta(_c(solver))
    varphi = get_varphi(_obj_val(solver), _x_lr(solver), _xl_r(solver), _xu_r(solver), _x_ur(solver), _mu(solver))
    kkt_error = max(_inf_pr(solver), _inf_du(solver), _inf_compl(solver))
    delta = kappa_1 * min(kappa_2, kkt_error)
    return is_filter_acceptable(_filter(solver), theta + delta, varphi + delta)
end

function update_barrier!(barrier::AbstractAdaptiveUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T
    old_mu = _mu(solver)
    progress = _check_progress(barrier, solver)
    # Update state of barrier algorithm
    if !barrier.free_mode
        if progress
            @trace(_logger(solver), "Moving adaptive barrier back to free mode.")
            barrier.free_mode = true
        else
            _update_monotone!(barrier, solver, sc)
        end
    else
        if !progress
            @trace(_logger(solver), "Moving adaptive barrier to monotone mode.")
            barrier.free_mode = false
            # Reset barrier parameter using current average complementarity
            set_mu!(solver, get_fixed_mu(solver, barrier))
        else
            @trace(_logger(solver), "Keeping adaptive barrier in free mode.")
        end
    end
    if barrier.free_mode
        set_mu!(solver, get_adaptive_mu(solver, barrier))
    end
    # Update tau and reset filter is barrier has been updated
    if _mu(solver) != old_mu
        set_tau!(solver, get_tau(_mu(solver), _opt(solver).tau_min))
        empty!(_filter(solver))
        push!(_filter(solver), (_theta_max(solver), -Inf))
    end
    return
end

function QualityFunctionUpdate(tol::T, barrier_tol_factor) where T
    return QualityFunctionUpdate{T}(; mu_min=min(1e-4, tol ) / (barrier_tol_factor + 1))
end

# Evaluate the linear quality function described in [Nocedal2009, Eq. (4.2)]
function _evaluate_quality_function(solver, sigma, step_aff, step_cen, res_dual, res_primal)
    n, m = _n(solver), _m(solver)
    nlb, nub = _nlb(solver), _nub(solver)
    tau = _tau(solver)
    d = _d(solver) # Load buffer
    # Δ(σ) = Δ_aff + σ Δ_cen
    full(d) .= full(step_aff) .+ sigma .* full(step_cen)
    # Primal step
    alpha_pr = get_alpha_max(
        primal(_x(solver)),
        primal(_xl(solver)),
        primal(_xu(solver)),
        primal(d),
        tau,
    )
    # Dual step
    alpha_du = get_alpha_z(
        _zl_r(solver),
        _zu_r(solver),
        dual_lb(d),
        dual_ub(d),
        tau,
    )
    # ||(X + αp ΔX - Xl) (Zl + αd ΔZl)||^2
    inf_compl_lb = mapreduce(
        (x, xl, dx, z, dz) -> ((x + alpha_pr * dx - xl) * (z + alpha_du * dz))^2,
        +,
        _x_lr(solver), _xl_r(solver), _dx_lr(solver), _zl_r(solver), dual_lb(d);
        init=0.0,
    )

    # ||(Xu - X - αp ΔX) (Zu + αd ΔZu)||^2
    inf_compl_ub = mapreduce(
        (x, xu, dx, z, dz) -> ((xu - x - alpha_pr * dx) * (z + alpha_du * dz))^2,
        +,
        _x_ur(solver), _xu_r(solver), _dx_ur(solver), _zu_r(solver), dual_ub(d);
        init=0.0,
    )
    # Primal infeasibility
    inf_pr = (m > 0) ? (1.0 - alpha_pr)^2 * res_primal^2 / m : 0.0
    # Dual infeasibility
    inf_du = (1.0 - alpha_du)^2 * res_dual^2 / n
    # Complementarity infeasibility
    inf_compl = (inf_compl_lb + inf_compl_ub ) / (nlb + nub)

    @debug(_logger(solver), @sprintf("sigma=%4.1e inf_pr=%4.2e inf_du=%4.2e inf_cc=%4.2e a_pr=%4.2e a_du=%4.2e", sigma, inf_pr, inf_du, inf_compl, alpha_pr, alpha_du))

    # Return quality function qL defined in Eq. (4.2)
    return inf_du + inf_pr + inf_compl
end

# Golden search algorithm to find a minimum of the linear quality function.
# The algorithm assumes the quality function is unimodular.
function _run_golden_search!(solver, barrier, sigma_lb, sigma_ub, step_aff, step_cen, res_primal, res_dual)
    gfac = 0.5 * (3.0 - sqrt(5.0))
    sigma_1, sigma_2 = sigma_lb, sigma_ub
    # Initiate Golden search
    phi_1 = _evaluate_quality_function(solver, sigma_1, step_aff, step_cen, res_primal, res_dual)
    phi_2 = _evaluate_quality_function(solver, sigma_2, step_aff, step_cen, res_primal, res_dual)
    sigma_1_in, sigma_2_in, phi_1_in, phi_2_in = sigma_1, sigma_2, phi_1, phi_2
    sigma_mid1 = sigma_lb + gfac * (sigma_ub - sigma_lb)
    sigma_mid2 = sigma_lb + (1.0 - gfac) * (sigma_ub - sigma_lb)
    phi_mid1 = _evaluate_quality_function(solver, sigma_mid1, step_aff, step_cen, res_primal, res_dual)
    phi_mid2 = _evaluate_quality_function(solver, sigma_mid2, step_aff, step_cen, res_primal, res_dual)
    # Run Golden search
    for i in 1:barrier.max_gs_iter
        if phi_mid1 > phi_mid2
            sigma_1 = sigma_mid1
            phi_1 = phi_mid1
            sigma_mid1 = sigma_mid2
            sigma_mid2 = sigma_1 + (1.0 - gfac) * (sigma_2 - sigma_1)
            phi_mid1 = phi_mid2
            phi_mid2 = _evaluate_quality_function(solver, sigma_mid2, step_aff, step_cen, res_primal, res_dual)
        else
            sigma_2 = sigma_mid2
            phi_2 = phi_mid2
            sigma_mid2 = sigma_mid1
            sigma_mid1 = sigma_1 + gfac * (sigma_2 - sigma_1)
            phi_mid1 = _evaluate_quality_function(solver, sigma_mid1, step_aff, step_cen, res_primal, res_dual)
            phi_mid2 = phi_mid1
        end
        if (sigma_2 - sigma_1 < barrier.sigma_tol * sigma_2)
            break
        end
    end
    # Compute final sigma
    sigma, phi = phi_mid1 < phi_mid2 ? (sigma_mid1, phi_mid1) : (sigma_mid2, phi_mid2)
    # Take into account that most the time the algorithm hasn't converged.
    if sigma_2 == sigma_2_in && phi_2_in < phi
        sigma = sigma_2_in
    elseif sigma_1 == sigma_1_in && phi_1_in < phi
        sigma = sigma_1_in
    end
    return sigma
end

function set_centering_aug_rhs!(solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, mu)
    px = primal(_p(solver))
    py = dual(_p(solver))
    pzl = dual_lb(_p(solver))
    pzu = dual_ub(_p(solver))
    px .= 0
    py .= 0
    pzl .= mu
    pzu .= -mu
    return
end

function get_adaptive_mu(solver::AbstractMadNLPSolver{T}, barrier::QualityFunctionUpdate{T}) where T
    # No inequality constraint: early return as barrier update is useless
    if _nlb(solver) + _nub(solver) == 0
        return barrier.mu_min
    end
    step_aff = __w3(solver) # buffer 1
    step_cen = __w4(solver) # buffer 2
    # Affine step
    set_aug_rhs!(solver, _kkt(solver), _c(solver), zero(T))
    # Get primal and dual infeasibility directly from the values in RHS p
    res_primal = norm(dual(_p(solver)))
    res_dual = norm(primal(_p(solver)))
    # Get approximate solution without iterative refinement
    copyto!(full(step_aff), full(_p(solver)))
    solve!(_kkt(solver), step_aff)
    # Get average complementarity
    mu = get_average_complementarity(solver)
    # Centering step
    set_centering_aug_rhs!(solver, _kkt(solver), mu)
    # NOTE(@anton) Ipopt also applies the dual infeasibility perturbation for some reason???
    dual_inf_perturbation!(primal(_p(solver)),_ind_llb(solver),_ind_uub(solver),mu,_opt(solver).kappa_d)
    # Get (again) approximate solution without iterative refinement
    copyto!(full(step_cen), full(_p(solver)))
    solve!(_kkt(solver), step_cen)
    # Refine the search interval using Ipopt's heuristics
    # First, check if sigma is greater than 1.
    phi1 = _evaluate_quality_function(solver, one(T), step_aff, step_cen, res_primal, res_dual)
    sigma_1m = one(T) - T(1e-4)
    phi1m = _evaluate_quality_function(solver, sigma_1m, step_aff, step_cen, res_primal, res_dual)
    # Restrict search interval
    if phi1m > phi1
        sigma_min = one(T)
        sigma_max = min(barrier.sigma_max, barrier.mu_max / mu)
    else
        sigma_min = max(barrier.sigma_min, barrier.mu_min / mu)
        sigma_max = min(max(sigma_min, sigma_1m), barrier.mu_max / mu)
    end
    # Run Golden-section search (assume the quality function is unimodal)
    sigma_opt = _run_golden_search!(solver, barrier, sigma_min, sigma_max, step_aff, step_cen, res_primal, res_dual)
    # Increment counter
    barrier.n_update += 1
    return clamp(sigma_opt * mu, barrier.mu_min, barrier.mu_max)
end

function LOQOUpdate(tol::T, barrier_tol_factor) where T
    return LOQOUpdate{T}(; mu_min=min(1e-4, tol ) / (barrier_tol_factor + 1))
end

function get_adaptive_mu(solver::AbstractMadNLPSolver{T}, barrier::LOQOUpdate{T}) where T
    # No inequality constraint: early return as barrier update is useless
    if _nlb(solver) + _nub(solver) == 0
        return barrier.mu_min
    end
    mu = get_average_complementarity(solver) # get average complementarity.
    ncc = _nlb(solver) + _nub(solver)
    min_cc = get_min_complementarity(solver)
    xi = min_cc/mu
    sigma = barrier.gamma*min((1-barrier.r)*((1-xi)/xi),2)^3
    mu = clamp(sigma * mu, barrier.mu_min, barrier.mu_max)
    return mu
end

