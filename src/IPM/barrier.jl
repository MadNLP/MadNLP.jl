
"""
    AbstractBarrierUpdate{T}

Abstraction used to implement the different rules to update the barrier parameter.
The barrier is updated using either a monotone rule or an adaptive rule.

"""
abstract type AbstractBarrierUpdate{T} end

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
        solver.x_lr,
        solver.xl_r,
        solver.zl_r,
        solver.xu_r,
        solver.x_ur,
        solver.zu_r,
        solver.mu,
        sc,
    )
    while (solver.mu > max(barrier.mu_min, solver.opt.tol/10)) &&
        (max(solver.inf_pr, solver.inf_du, inf_compl_mu) <= solver.opt.barrier_tol_factor*solver.mu)
        mu_new = get_mu(
            solver.mu,
            barrier.mu_min,
            barrier.mu_linear_decrease_factor,
            barrier.mu_superlinear_decrease_power,
            solver.opt.tol,
        )
        inf_compl_mu = get_inf_compl(
            solver.x_lr,
            solver.xl_r,
            solver.zl_r,
            solver.xu_r,
            solver.x_ur,
            solver.zu_r,
            solver.mu,
            sc,
        )
        solver.tau = get_tau(solver.mu, solver.opt.tau_min)
        solver.mu = mu_new
        empty!(solver.filter)
        push!(solver.filter, (solver.theta_max, -Inf))
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
    RR = solver.RR
    inf_compl_mu_R = get_inf_compl_R(
        solver.x_lr,
        solver.xl_r,
        solver.zl_r,
        solver.xu_r,
        solver.x_ur,
        solver.zu_r,
        RR.pp,
        RR.zp,
        RR.nn,
        RR.zn,
        RR.mu_R,
        sc,
    )
    while RR.mu_R >= barrier.mu_min &&
        max(RR.inf_pr_R,RR.inf_du_R,inf_compl_mu_R) <= solver.opt.barrier_tol_factor*RR.mu_R
        RR.mu_R = get_mu(
            RR.mu_R,
            barrier.mu_min,
            barrier.mu_linear_decrease_factor,
            barrier.mu_superlinear_decrease_power,
            solver.opt.tol,
        )
        inf_compl_mu_R = get_inf_compl_R(
            solver.x_lr,
            solver.xl_r,
            solver.zl_r,
            solver.xu_r,
            solver.x_ur,
            solver.zu_r,
            RR.pp,
            RR.zp,
            RR.nn,
            RR.zn,
            RR.mu_R,
            sc,
        )
        RR.tau_R= max(solver.opt.tau_min,1-RR.mu_R)
        RR.zeta = sqrt(RR.mu_R)
        empty!(RR.filter)
        push!(RR.filter,(solver.theta_max,-Inf))
    end
    return
end

"""
    MonotoneUpdate{T} <: AbstractBarrierUpdate{T}

Update the barrier parameter using the classical Fiacco-McCormick monotone rule.

"""
@kwdef mutable struct MonotoneUpdate{T} <: AbstractBarrierUpdate{T}
    mu_init::T = 1e-1
    mu_min::T = 1e-11
    mu_superlinear_decrease_power::T = 1.5
    mu_linear_decrease_factor::T = .2
end
function MonotoneUpdate(tol::T, barrier_tol_factor) where T
    return MonotoneUpdate{T}(; mu_min=min(1e-4, tol ) / (barrier_tol_factor + 1))
end

function update_barrier!(barrier::MonotoneUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T
    _update_monotone!(barrier, solver, sc)
end

#=
    Adaptive updates
=#

"""
    AbstractAdaptiveUpdate{T} <: AbstractBarrierUpdate{T}

Abstraction used to implement the adaptive barrier updates described in [Nocedal2009].

## References
[Nocedal2009] Nocedal, J., Wächter, A., & Waltz, R. A. (2009).
Adaptive barrier update strategies for nonlinear interior methods.
SIAM Journal on Optimization, 19(4), 1674-1693.

"""
abstract type AbstractAdaptiveUpdate{T} <: AbstractBarrierUpdate{T} end

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
    theta = get_theta(solver.c)
    varphi = get_varphi(solver.obj_val, solver.x_lr, solver.xl_r, solver.xu_r, solver.x_ur, solver.mu)
    kkt_error = max(solver.inf_pr, solver.inf_du, solver.inf_compl)
    delta = kappa_1 * min(kappa_2, kkt_error)
    return is_filter_acceptable(solver.filter, theta + delta, varphi + delta)
end

function update_barrier!(barrier::AbstractAdaptiveUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T
    old_mu = solver.mu
    progress = _check_progress(barrier, solver)
    # Update state of barrier algorithm
    if !barrier.free_mode
        if progress
            @trace(solver.logger, "Moving adaptive barrier back to free mode.")
            barrier.free_mode = true
        else
            _update_monotone!(barrier, solver, sc)
        end
    else
        if !progress
            @trace(solver.logger, "Moving adaptive barrier to monotone mode.")
            barrier.free_mode = false
            # Reset barrier parameter using current average complementarity
            solver.mu = get_fixed_mu(solver, barrier)
        else
            @trace(solver.logger, "Keeping adaptive barrier in free mode.")
        end
    end
    if barrier.free_mode
        solver.mu = get_adaptive_mu(solver, barrier)
    end
    # Update tau and reset filter is barrier has been updated
    if solver.mu != old_mu
        solver.tau = get_tau(solver.mu, solver.opt.tau_min)
        empty!(solver.filter)
        push!(solver.filter, (solver.theta_max, -Inf))
    end
    return
end

"""
    QualityFunctionUpdate{T} <: AbstractAdaptiveUpdate{T}

Find the barrier parameter using a quality function encoding the ℓ1-norm of the KKT violations.
At each IPM iteration, the minimum of the quality function is found using a Golden search algorithm.

If no sufficient progress is made, the barrier fallbacks to a monotone rule.

## References
The algorithm is described in [Nocedal2009, Section 4].

"""
@kwdef mutable struct QualityFunctionUpdate{T} <: AbstractAdaptiveUpdate{T}
    mu_init::T = 1e-1
    mu_min::T = 1e-11
    mu_max::T = 1e5
    sigma_min::T = 1e-6
    sigma_max::T = 1e2
    sigma_tol::T = 1e-2
    gamma::T = 1.0
    max_gs_iter::Int = 8
    # For non-free mode
    mu_superlinear_decrease_power::T = 1.5
    mu_linear_decrease_factor::T = .2
    free_mode::Bool = true
    globalization::Bool = true
    n_update::Int = 0
end
function QualityFunctionUpdate(tol::T, barrier_tol_factor) where T
    return QualityFunctionUpdate{T}(; mu_min=min(1e-4, tol ) / (barrier_tol_factor + 1))
end

# Evaluate the linear quality function described in [Nocedal2009, Eq. (4.2)]
function _evaluate_quality_function(solver, sigma, step_aff, step_cen, res_dual, res_primal)
    n, m = solver.n, solver.m
    nlb, nub = solver.nlb, solver.nub
    tau = solver.tau
    d = solver.d # Load buffer
    # Δ(σ) = Δ_aff + σ Δ_cen
    full(d) .= full(step_aff) .+ sigma .* full(step_cen)
    # Primal step
    alpha_pr = get_alpha_max(
        primal(solver.x),
        primal(solver.xl),
        primal(solver.xu),
        primal(d),
        tau,
    )
    # Dual step
    alpha_du = get_alpha_z(
        solver.zl_r,
        solver.zu_r,
        dual_lb(d),
        dual_ub(d),
        tau,
    )
    # ||(X + αp ΔX - Xl) (Zl + αd ΔZl)||^2
    inf_compl_lb = mapreduce(
        (x, xl, dx, z, dz) -> ((x + alpha_pr * dx - xl) * (z + alpha_du * dz))^2,
        +,
        solver.x_lr, solver.xl_r, solver.dx_lr, solver.zl_r, dual_lb(d);
        init=0.0,
    )

    # ||(Xu - X - αp ΔX) (Zu + αd ΔZu)||^2
    inf_compl_ub = mapreduce(
        (x, xu, dx, z, dz) -> ((xu - x - alpha_pr * dx) * (z + alpha_du * dz))^2,
        +,
        solver.x_ur, solver.xu_r, solver.dx_ur, solver.zu_r, dual_ub(d);
        init=0.0,
    )
    # Primal infeasibility
    inf_pr = (m > 0) ? (1.0 - alpha_pr)^2 * res_primal^2 / m : 0.0
    # Dual infeasibility
    inf_du = (1.0 - alpha_du)^2 * res_dual^2 / n
    # Complementarity infeasibility
    inf_compl = (inf_compl_lb + inf_compl_ub ) / (nlb + nub)

    @debug(solver.logger, @sprintf("sigma=%4.1e inf_pr=%4.2e inf_du=%4.2e inf_cc=%4.2e a_pr=%4.2e a_du=%4.2e", sigma, inf_pr, inf_du, inf_compl, alpha_pr, alpha_du))

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
    px = primal(solver.p)
    py = dual(solver.p)
    pzl = dual_lb(solver.p)
    pzu = dual_ub(solver.p)
    px .= 0
    py .= 0
    pzl .= mu
    pzu .= -mu
    return
end

function get_adaptive_mu(solver::AbstractMadNLPSolver{T}, barrier::QualityFunctionUpdate{T}) where T
    # No inequality constraint: early return as barrier update is useless
    if solver.nlb + solver.nub == 0
        return barrier.mu_min
    end
    step_aff = solver._w3 # buffer 1
    step_cen = solver._w4 # buffer 2
    # Affine step
    set_aug_rhs!(solver, solver.kkt, solver.c, zero(T))
    # Get primal and dual infeasibility directly from the values in RHS p
    res_primal = norm(dual(solver.p))
    res_dual = norm(primal(solver.p))
    # Get approximate solution without iterative refinement
    copyto!(full(step_aff), full(solver.p))
    solve!(solver.kkt, step_aff)
    # Get average complementarity
    mu = get_average_complementarity(solver)
    # Centering step
    set_centering_aug_rhs!(solver, solver.kkt, mu)
    # NOTE(@anton) Ipopt also applies the dual infeasibility perturbation for some reason???
    dual_inf_perturbation!(primal(solver.p),solver.ind_llb,solver.ind_uub,mu,solver.opt.kappa_d)
    # Get (again) approximate solution without iterative refinement
    copyto!(full(step_cen), full(solver.p))
    solve!(solver.kkt, step_cen)
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

"""
    LOQOUpdate{T} <: AbstractAdaptiveUpdate{T}

Find the barrier parameter using the rule used in the LOQO solver.
The rule is explicited in [Nocedal2009, Eq (3.6)].

If no sufficient progress is made, the barrier fallbacks to a monotone rule.

## References
The algorithm is described in [Nocedal2009, Section 3].

"""
@kwdef mutable struct LOQOUpdate{T} <: AbstractAdaptiveUpdate{T}
    mu_init::T = 1e-1
    mu_min::T = 1e-11
    mu_max::T = 1e5
    gamma::T = 0.1 # scale factor
    r::T = .95 # Steplength param
    mu_superlinear_decrease_power::T = 1.5
    mu_linear_decrease_factor::T = .2
    free_mode::Bool = true
    globalization::Bool = true
end
function LOQOUpdate(tol::T, barrier_tol_factor) where T
    return LOQOUpdate{T}(; mu_min=min(1e-4, tol ) / (barrier_tol_factor + 1))
end

function get_adaptive_mu(solver::AbstractMadNLPSolver{T}, barrier::LOQOUpdate{T}) where T
    # No inequality constraint: early return as barrier update is useless
    if solver.nlb + solver.nub == 0
        return barrier.mu_min
    end
    mu = get_average_complementarity(solver) # get average complementarity.
    ncc = solver.nlb + solver.nub
    min_cc = get_min_complementarity(solver)
    xi = min_cc/mu
    sigma = barrier.gamma*min((1-barrier.r)*((1-xi)/xi),2)^3
    mu = clamp(sigma * mu, barrier.mu_min, barrier.mu_max)
    return mu
end

