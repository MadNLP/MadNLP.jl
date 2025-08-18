

abstract type AbstractBarrierUpdate{T} end

@kwdef struct MonotoneUpdate{T} <: AbstractBarrierUpdate{T}
    mu_init::T = 1e-1
    mu_min::T = 1e-11
    mu_superlinear_decrease_power::T = 1.5
    mu_linear_decrease_factor::T = .2
end

function update_barrier!(barrier::MonotoneUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T
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
end


@kwdef mutable struct AdaptiveUpdate{T} <: AbstractBarrierUpdate{T}
    mu_init::T = 1e-1
    mu_min::T = 1e-11
    mu_max::T = 1e5
    sigma_min::T = 1e-6
    sigma_max::T = 1e2
    sigma_tol::T = 1e-2
    gamma::T = 1.0
    free_mode::Bool = true
    max_gs_iter::Int = 8

    # For non-free mode (also temporarily for robust solve :P )
    mu_superlinear_decrease_power::T = 1.5
    mu_linear_decrease_factor::T = .2
end

function get_fixed_mu(solver::AbstractMadNLPSolver{T}, barrier::AdaptiveUpdate{T}) where T
    mu = T(0.8) * get_average_complementarity(solver)
    return clamp(mu, barrier.mu_min, mu_max)
end

function _evaluate_quality_function(solver, sigma, step_aff, step_cen, res_dual, res_primal)
    n, m = solver.n, solver.m
    nlb, nub = solver.nlb, solver.nub
    tau = solver.tau
    d = solver.d # Load buffer

    # Δ(σ) = Δ(0) + σ (Δ(1) - Δ(0))
    full(d) .= full(step_aff) .+ sigma .* (full(step_cen) .- full(step_aff))
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

    # (x + αp Δx - xl)ᵀ (zl + αd Δzl)
    inf_compl_lb = mapreduce(
        (x, xl, dx, z, dz) -> ((x + alpha_pr * dx - xl) * (z + alpha_du * dz))^2,
        +,
        solver.x_lr, solver.xl_r, solver.dx_lr, solver.zl_r, dual_lb(d);
        init=0.0,
    )
    # (xu - x - αp Δx)ᵀ (zu + αd Δzu)
    inf_compl_ub = mapreduce(
        (x, xu, dx, z, dz) -> ((xu - x - alpha_pr * dx) * (z + alpha_du * dz))^2,
        +,
        solver.x_ur, solver.xu_r, solver.dx_ur, solver.zu_r, dual_ub(d);
        init=0.0,
    )

    # Primal infeasibility
    inf_pr = (1.0 - alpha_pr)^2 * res_primal^2 / n
    # Dual infeasibility
    inf_du = (1.0 - alpha_du)^2 * res_dual^2 / m
    # Complementarity infeasibility
    inf_compl = (inf_compl_lb + inf_compl_ub ) / (nlb + nub)

    # Quality function qL defined in Eq. (4.2)
    return inf_du + inf_pr + inf_compl
end

function _run_golden_search!(solver, barrier, sigma_lb, sigma_ub, step_aff, step_cen, res_primal, res_dual)
    gfac = 0.5 * (3.0 - sqrt(5.0))

    sigma_1, sigma_2 = sigma_lb, sigma_ub
    phi_1 = _evaluate_quality_function(solver, sigma_1, step_aff, step_cen, res_primal, res_dual)
    phi_2 = _evaluate_quality_function(solver, sigma_2, step_aff, step_cen, res_primal, res_dual)

    sigma_mid1 = sigma_lb + gfac * (sigma_ub - sigma_lb)
    sigma_mid2 = sigma_lb + (1.0 - gfac) * (sigma_ub - sigma_lb)
    phi_mid1 = _evaluate_quality_function(solver, sigma_mid1, step_aff, step_cen, res_primal, res_dual)
    phi_mid2 = _evaluate_quality_function(solver, sigma_mid2, step_aff, step_cen, res_primal, res_dual)

    # Golden search
    for i in 1:barrier.max_gs_iter
        if phi_mid1 > phi_mid2
            sigma_1 = sigma_mid1
            phi_1 = phi_mid1
            sigma_mid1 = sigma_mid2
            sigma_mid2 = sigma_1 + (1.0 - gfac) * (sigma_2 - sigma_1)
            phi_mid2 = _evaluate_quality_function(solver, sigma_mid2, step_aff, step_cen, res_primal, res_dual)
        else
            sigma_2 = sigma_mid2
            phi_2 = phi_mid2
            sigma_mid2 = sigma_mid1
            sigma_mid1 = sigma_1 + gfac * (sigma_2 - sigma_1)
            phi_mid1 = _evaluate_quality_function(solver, sigma_mid1, step_aff, step_cen, res_primal, res_dual)
        end

        if sigma_2 - sigma_1 < barrier.sigma_tol * sigma_2
            break
        end
    end
    # Compute final sigma
    sigma, phi = phi_mid1 < phi_mid2 ? (sigma_mid1, phi_mid1) : (sigma_mid2, phi_mid2)
    return sigma
end

function set_cen_aug_rhs!(solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, mu)
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

function get_adaptive_mu(solver::AbstractMadNLPSolver, barrier::AdaptiveUpdate)
    linear_solver = solver.kkt.linear_solver
    step_aff = solver._w1 # buffer 1
    step_cen = solver._w2 # buffer 2

    # Affine step
    set_aug_rhs!(solver, solver.kkt, solver.c, 0.0)
    # Get primal and dual infeasibility directly 1from the values in RHS p
    res_primal = norm(dual(solver.p))
    res_dual = norm(primal(solver.p))

    # Get approximate solution without iterative refinement
    copyto!(full(step_aff), full(solver.p))
    solve!(linear_solver, full(step_aff))

    # Get average complementarity
    mu = get_average_complementarity(solver)
    # Centering step
    set_cen_aug_rhs!(solver, solver.kkt, mu)
    # NOTE(@anton) Ipopt also applies the dual infeasibility perturbation for some reason???
    dual_inf_perturbation!(primal(solver.p),solver.ind_llb,solver.ind_uub,mu,solver.opt.kappa_d)
    # Get (again) approximate solution without iterative refinement
    copyto!(full(step_cen), full(solver.p))
    solve!(linear_solver, full(step_cen))

    # Refine the search interval using Ipopt's heuristics
    # First, check if sigma is greater than 1.
    phi1 = _evaluate_quality_function(solver, 1.0, step_aff, step_cen, res_primal, res_dual)
    sigma_1m = 1.0 - 1e-4
    phi1m = _evaluate_quality_function(solver, sigma_1m, step_aff, step_cen, res_primal, res_dual)
    # Restrict search interval
    if phi1m > phi1
        sigma_min = 1.0
        sigma_max = min(barrier.sigma_max, barrier.mu_max / mu)
    else
        sigma_min = max(barrier.sigma_min, barrier.mu_min / mu)
        sigma_max = min(max(sigma_min, sigma_1m), barrier.mu_max / mu)
    end

    # Run Golden-section search (assume the quality function is unimodal)
    sigma_opt = _run_golden_search!(solver, barrier, sigma_min, sigma_max, step_aff, step_cen, res_primal, res_dual)
    return sigma_opt * mu
end

function update_barrier!(barrier::AdaptiveUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T
    kappa_1 = T(1e-5)
    kappa_2 = T(1.0)

    # TODO: implement fixed mode
    mu = get_adaptive_mu(solver, barrier)
    # TODO: check sufficient progress using filter line-search
    # theta = NaN
    # varphi = NaN
    # delta = NaN
    # progress = is_filter_acceptable(solver.filter, theta + delta, varphi + delta)

    # Just a sketch for now
    # if barrier.free_mode
    #     if progress
    #         mu = get_adaptive_mu(solver.mu)
    #     else
    #         barrier.free_mode = false
    #         # Get initial fixed barrier
    #         mu = barrier_fixed_mu(barrier)
    #     end
    # else
    #     if progress
    #         barrier.free_mode = true
    #     else
    #         # Monotone update
    #         # TODO
    #     end
    # end

    # Update tau
    solver.mu = max(mu, barrier.mu_min)
    solver.tau = get_tau(solver.mu, solver.opt.tau_min)
    # Reset filter line-search
    empty!(solver.filter)
    push!(solver.filter, (solver.theta_max, -Inf))
end

@kwdef mutable struct LOQOUpdate{T} <: AbstractBarrierUpdate{T}
    mu_init::T = 1e-1
    mu_min::T = 1e-11
    gamma::T = 0.1 # scale factor
    r::T = .95 # Steplength param
    # Temporarily for robust solve
    mu_superlinear_decrease_power::T = 1.5
    mu_linear_decrease_factor::T = .2
end

function update_barrier!(barrier::LOQOUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T
    mu = get_average_complementarity(solver) # get average complementarity.
    ncc = solver.nlb + solver.nub
    min_cc = get_min_complementarity(solver)
    xi = min_cc/mu
    sigma = barrier.gamma*min((1-barrier.r)*((1-xi)/xi),2)^3

    solver.mu = max(barrier.mu_min,sigma*mu)
    # TODO(@anton): Hmmmm does this make sense, we essentially throw out filter always
    empty!(solver.filter)
    push!(solver.filter, (solver.theta_max, -Inf))
end
