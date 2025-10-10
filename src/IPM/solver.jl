"""
    madnlp(model::AbstractNLPModel; options...)

Build a [`MadNLPSolver`](@ref) and solve it using
the interior-point method. Return the solution
as a [`MadNLPExecutionStats`](@ref).

"""
function madnlp(model::AbstractNLPModel; kwargs...)
    solver = MadNLPSolver(model;kwargs...)
    return solve!(solver)
end

solve!(nlp::AbstractNLPModel, solver::AbstractMadNLPSolver; kwargs...) = solve!(
    nlp, solver, MadNLPExecutionStats(solver);
    kwargs...)
solve!(solver::AbstractMadNLPSolver; kwargs...) = solve!(
    get_nlp(solver), solver;
    kwargs...)


function initialize!(solver::AbstractMadNLPSolver{T}) where T

    nlp = get_nlp(solver)
    opt = get_opt(solver)

    # Initializing variables
    @trace(get_logger(solver),"Initializing variables.")
    initialize!(
        get_cb(solver),
        get_x(solver),
        get_xl(solver),
        get_xu(solver),
        get_y(solver),
        get_rhs(solver),
        get_ind_ineq(solver);
        tol=opt.bound_relax_factor,
        bound_push=opt.bound_push,
        bound_fac=opt.bound_fac,
    )
    fill!(get_jacl(solver), zero(T))
    fill!(get_zl_r(solver), one(T))
    fill!(get_zu_r(solver), one(T))

    # Initializing scaling factors
    if opt.nlp_scaling
        set_scaling!(
            get_cb(solver),
            get_x(solver),
            get_xl(solver),
            get_xu(solver),
            get_y(solver),
            get_rhs(solver),
            get_ind_ineq(solver),
            opt.nlp_scaling_max_gradient
        )
    end

    # Initializing KKT system
    initialize!(get_kkt(solver))

    # Initializing jacobian and gradient
    eval_jac_wrapper!(solver, get_kkt(solver), get_x(solver))
    eval_grad_f_wrapper!(solver, get_f(solver),get_x(solver))


    @trace(get_logger(solver),"Initializing constraint duals.")
    if !get_opt(solver).dual_initialized
        initialize_dual(solver, opt.dual_initialization_method)
    end

    # Initializing
    set_obj_val!(solver, eval_f_wrapper(solver, get_x(solver)))
    eval_cons_wrapper!(solver, get_c(solver), get_x(solver))
    eval_lag_hess_wrapper!(solver, get_kkt(solver), get_x(solver), get_y(solver))

    theta = get_theta(get_c(solver))
    set_theta_max!(solver, T(1e4)*max(one(T),theta))
    set_theta_min!(solver, T(1e-4)*max(one(T),theta))
    set_mu!(solver, get_opt(solver).barrier.mu_init)
    set_tau!(solver, max(get_opt(solver).tau_min,one(T)-get_opt(solver).barrier.mu_init))
    push!(get_filter(solver), (get_theta_max(solver),-T(Inf)))

    return REGULAR
end

abstract type DualInitializeOptions end
struct DualInitializeSetZero <: DualInitializeOptions end
struct DualInitializeLeastSquares <: DualInitializeOptions end

function initialize_dual(solver::AbstractMadNLPSolver{T}, ::Type{DualInitializeSetZero}) where T
    fill!(get_y(solver), zero(T))
end
function initialize_dual(solver::AbstractMadNLPSolver{T}, ::Type{DualInitializeLeastSquares}) where T
    set_initial_rhs!(solver, get_kkt(solver))
    factorize_wrapper!(solver)
    is_solved = solve_refine_wrapper!(
        get_d(solver), solver, get_p(solver), get__w4(solver)
    )
    if !is_solved || (norm(dual(get_d(solver)), T(Inf)) > get_opt(solver).constr_mult_init_max)
        fill!(get_y(solver), zero(T))
    else
        copyto!(get_y(solver), dual(get_d(solver)))
    end
end

function reinitialize!(solver::AbstractMadNLPSolver)
    variable(get_x(solver)) .= get_x0(get_nlp(solver))

    set_obj_val!(solver, eval_f_wrapper(solver, get_x(solver)))
    eval_grad_f_wrapper!(solver, get_f(solver), get_x(solver))
    eval_cons_wrapper!(solver, get_c(solver), get_x(solver))
    eval_jac_wrapper!(solver, get_kkt(solver), get_x(solver))
    eval_lag_hess_wrapper!(solver, get_kkt(solver), get_x(solver), get_y(solver))

    theta = get_theta(get_c(solver))
    set_theta_max!(solver, 1e4*max(1,theta))
    set_theta_min!(solver, 1e-4*max(1,theta))
    set_mu!(solver, get_opt(solver).barrier.mu_init)
    set_tau!(solver, max(get_opt(solver).tau_min,1-get_opt(solver).barrier.mu_init))
    empty!(get_filter(solver))
    push!(get_filter(solver), (get_theta_max(solver),-Inf))

    return REGULAR
end

# major loops ---------------------------------------------------------
function solve!(
    nlp::AbstractNLPModel,
    solver::AbstractMadNLPSolver,
    stats::MadNLPExecutionStats;
    x = nothing, y = nothing,
    zl = nothing, zu = nothing,
    kwargs...
        )

    if x != nothing
        full(get_x(solver))[1:get_nvar(nlp)] .= x
    end
    if y != nothing
        get_y(solver)[1:get_ncon(nlp)] .= y
    end
    if zl != nothing
        full(get_zl(solver))[1:get_nvar(nlp)] .= zl
    end
    if zu != nothing
        full(get_zu(solver))[1:get_nvar(nlp)] .= zu
    end

    if !isempty(kwargs)
        @warn(get_logger(solver),"The options set during resolve may not have an effect")
        set_options!(get_opt(solver), kwargs)
    end

    try
        if get_status(solver) == INITIAL
            @notice(get_logger(solver),"This is $(introduce()), running with $(introduce(get_kkt(solver).linear_solver))\n")
            print_init(solver)
            set_status!(solver, initialize!(solver))
        else # resolving the problem
            set_status!(solver, reinitialize!(solver))
        end

        while get_status(solver) >= REGULAR
            get_status(solver) == REGULAR && (set_status!(solver, regular!(solver)))
            get_status(solver) == RESTORE && (set_status!(solver, restore!(solver)))
            get_status(solver) == ROBUST && (set_status!(solver, robust!(solver)))
        end
    catch e
        if e isa InvalidNumberException
            if e.callback == :obj
                set_status!(solver, INVALID_NUMBER_OBJECTIVE)
            elseif e.callback == :grad
                set_status!(solver, INVALID_NUMBER_GRADIENT)
            elseif e.callback == :cons
                set_status!(solver, INVALID_NUMBER_CONSTRAINTS)
            elseif e.callback == :jac
                set_status!(solver, INVALID_NUMBER_JACOBIAN)
            elseif e.callback == :hess
                set_status!(solver, INVALID_NUMBER_HESSIAN_LAGRANGIAN)
            else
                set_status!(solver, INVALID_NUMBER_DETECTED)
            end
        elseif e isa NotEnoughDegreesOfFreedomException
            set_status!(solver, NOT_ENOUGH_DEGREES_OF_FREEDOM)
        elseif e isa LinearSolverException
            set_status!(solver, ERROR_IN_STEP_COMPUTATION;)
            get_opt(solver).rethrow_error && rethrow(e)
        elseif e isa InterruptException
            set_status!(solver, USER_REQUESTED_STOP)
            get_opt(solver).rethrow_error && rethrow(e)
        else
            set_status!(solver, INTERNAL_ERROR)
            get_opt(solver).rethrow_error && rethrow(e)
        end
    finally
        get_cnt(solver).total_time = time() - get_cnt(solver).start_time
        if !(get_status(solver) < SOLVE_SUCCEEDED)
            print_summary(solver)
        end
        @notice(get_logger(solver),"$(Base.text_colors[color_status(get_status(solver))])EXIT: $(get_status_output(get_status(solver), get_opt(solver)))$(Base.text_colors[:normal])")
        get_opt(solver).disable_garbage_collector &&
            (GC.enable(true); @warn(get_logger(solver),"Julia garbage collector is turned back on"))
        finalize(get_logger(solver))

        update!(stats,solver)
    end


    return stats
end

color_status(status::Status) =
    status <= SOLVE_SUCCEEDED ? :green :
    status <= SOLVED_TO_ACCEPTABLE_LEVEL ? :blue : :red


function regular!(solver::AbstractMadNLPSolver{T}) where T
    while true
        eval_for_next_iter!(solver)

        print_iter(solver)

        # evaluate termination criteria
        (status = evaluate_termination_criteria!(solver)) == REGULAR || return status

        # evaluate Hessian
        evaluate_hessian!(solver)

        # update the barrier parameter
        update_homotopy_parameters!(solver)

        # compute the newton step
        (status = compute_newton_step!(solver)) == REGULAR || return status

        # line search
        (status = line_search!(solver)) == LINESEARCH_SUCCEEDED || return status

        update_variables!(solver)

        _cnt(solver).k+=1
        @trace(_logger(solver),"Proceeding to the next interior point iteration.")
    end
end


function restore!(solver::AbstractMadNLPSolver{T}) where T
    set_del_w!(solver, zero(T))
    # Backup the previous primal iterate
    copyto!(primal(get__w1(solver)), full(get_x(solver)))
    copyto!(dual(get__w1(solver)), get_y(solver))
    copyto!(dual(get__w2(solver)), get_c(solver))

    F = get_F(
        get_c(solver),
        primal(get_f(solver)),
        primal(get_zl(solver)),
        primal(get_zu(solver)),
        get_jacl(solver),
        get_x_lr(solver),
        get_xl_r(solver),
        get_zl_r(solver),
        get_xu_r(solver),
        get_x_ur(solver),
        get_zu_r(solver),
        get_mu(solver),
    )
    set_alpha_z!(solver, zero(T))
    set_ftype!(solver, "R")
    while true
        alpha_max = get_alpha_max(
            primal(get_x(solver)),
            primal(get_xl(solver)),
            primal(get_xu(solver)),
            primal(get_d(solver)),
            get_tau(solver),
        )
        set_alpha!(solver,min(
            alpha_max,
            get_alpha_z(get_zl_r(solver),get_zu_r(solver),dual_lb(get_d(solver)),dual_ub(get_d(solver)),get_tau(solver)),
        ))

        axpy!(get_alpha(solver), primal(get_d(solver)), full(get_x(solver)))
        axpy!(get_alpha(solver), dual(get_d(solver)), get_y(solver))
        get_zl_r(solver) .+= get_alpha(solver) .* dual_lb(get_d(solver))
        get_zu_r(solver) .+= get_alpha(solver) .* dual_ub(get_d(solver))

        eval_cons_wrapper!(solver,get_c(solver),get_x(solver))
        eval_grad_f_wrapper!(solver,get_f(solver),get_x(solver))
        set_obj_val!(solver, eval_f_wrapper(solver,get_x(solver)))

        !get_opt(solver).jacobian_constant && eval_jac_wrapper!(solver,get_kkt(solver),get_x(solver))
        jtprod!(get_jacl(solver),get_kkt(solver),get_y(solver))

        F_trial = get_F(
            get_c(solver),
            primal(get_f(solver)),
            primal(get_zl(solver)),
            primal(get_zu(solver)),
            get_jacl(solver),
            get_x_lr(solver),
            get_xl_r(solver),
            get_zl_r(solver),
            get_xu_r(solver),
            get_x_ur(solver),
            get_zu_r(solver),
            get_mu(solver),
        )
        if F_trial > get_opt(solver).soft_resto_pderror_reduction_factor*F
            copyto!(primal(get_x(solver)), primal(get__w1(solver)))
            copyto!(get_y(solver), dual(get__w1(solver)))
            copyto!(get_c(solver), dual(get__w2(solver))) # backup the previous primal iterate
            return ROBUST
        end

        adjust_boundary!(get_x_lr(solver),get_xl_r(solver),get_x_ur(solver),get_xu_r(solver),get_mu(solver))

        F = F_trial

        theta = get_theta(get_c(solver))
        varphi= get_varphi(get_obj_val(solver),get_x_lr(solver),get_xl_r(solver),get_xu_r(solver),get_x_ur(solver),get_mu(solver))

        get_cnt(solver).k+=1

        is_filter_acceptable(get_filter(solver),theta,varphi) ? (return REGULAR) : (get_cnt(solver).t+=1)
        get_cnt(solver).k>=get_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-get_cnt(solver).start_time>=get_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED


        sd = get_sd(get_y(solver),get_zl_r(solver),get_zu_r(solver),get_opt(solver).s_max)
        sc = get_sc(get_zl_r(solver),get_zu_r(solver),get_opt(solver).s_max)
        set_inf_pr!(solver, get_inf_pr(get_c(solver)))
        set_inf_du!(solver, get_inf_du(
            primal(get_f(solver)),
            primal(get_zl(solver)),
            primal(get_zu(solver)),
            get_jacl(solver),
            sd,
        ))

        set_inf_compl!(solver, get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),zero(T),sc))
        set_inf_compl_mu!(solver, get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),get_mu(solver),sc))
        print_iter(solver)

        !get_opt(solver).hessian_constant && eval_lag_hess_wrapper!(solver,get_kkt(solver),get_x(solver),get_y(solver))
        set_aug_diagonal!(get_kkt(solver),solver)
        set_aug_rhs!(solver, get_kkt(solver), get_c(solver), get_mu(solver))

        dual_inf_perturbation!(primal(get_p(solver)),get_ind_llb(solver),get_ind_uub(solver),get_mu(solver),get_opt(solver).kappa_d)
        factorize_wrapper!(solver)
        solve_refine_wrapper!(
            get_d(solver), solver, get_p(solver), get__w4(solver)
        )

        set_ftype!(solver, "f")
    end
end

function robust!(solver::AbstractMadNLPSolver{T}) where T
    initialize_robust_restorer!(solver)
    RR = get_RR(solver)
    while true
        # evaluate termination criteria
        eval_for_next_iter_RR!(solver)

        print_iter(solver;is_resto=true)

        (status = evaluate_termination_criteria_RR!(solver)) == ROBUST || return status

        # update the barrier parameter
        update_homotopy_parameters_RR!(solver)

        # compute the newton step
        (status = compute_newton_step_RR!(solver)) == ROBUST || return status

        # filter start
        (status = line_search_RR!(solver)) == LINESEARCH_SUCCEEDED || return status

        # update variables
        update_variables_RR!(solver)

        # compeleted an iteration
        _cnt(solver).k+=1
        _cnt(solver).t+=1
        
        # check if going back to regular phase
        status = check_restoration_successful!(solver)
        if status == REGULAR
            return_from_restoration!(solver)
            return REGULAR
        end

        @trace(_logger(solver),"Proceeding to the next restoration phase iteration.")
    end
end

function second_order_correction(solver::AbstractMadNLPSolver,alpha_max,theta,varphi,
                                 theta_trial,varphi_d,switching_condition::Bool)
    @trace(get_logger(solver),"Second-order correction started.")

    wx = primal(get__w1(solver))
    wy = dual(get__w1(solver))
    copyto!(wy, get_c_trial(solver))
    axpy!(alpha_max, get_c(solver), wy)

    theta_soc_old = theta_trial
    for p=1:get_opt(solver).max_soc
        # compute second order correction
        set_aug_rhs!(solver, get_kkt(solver), wy, get_mu(solver))
        dual_inf_perturbation!(
            primal(get_p(solver)),
            get_ind_llb(solver),get_ind_uub(solver),get_mu(solver),get_opt(solver).kappa_d,
        )
        solve_refine_wrapper!(
            get__w1(solver), solver, get_p(solver), get__w4(solver)
        )
        alpha_soc = get_alpha_max(
            primal(get_x(solver)),
            primal(get_xl(solver)),
            primal(get_xu(solver)),
            wx,get_tau(solver)
        )

        copyto!(primal(get_x_trial(solver)), primal(get_x(solver)))
        axpy!(alpha_soc, wx, primal(get_x_trial(solver)))
        eval_cons_wrapper!(solver, get_c_trial(solver), get_x_trial(solver))
        set_obj_val_trial!(solver, eval_f_wrapper(solver, get_x_trial(solver)))

        theta_soc = get_theta(get_c_trial(solver))
        varphi_soc= get_varphi(get_obj_val_trial(solver),get_x_trial_lr(solver),get_xl_r(solver),get_xu_r(solver),get_x_trial_ur(solver),get_mu(solver))

        !is_filter_acceptable(get_filter(solver),theta_soc,varphi_soc) && break

        if theta <=get_theta_min(solver) && switching_condition
            # Case I
            if is_armijo(varphi_soc,varphi,get_opt(solver).eta_phi,get_alpha(solver),varphi_d)
                @trace(get_logger(solver),"Step in second order correction accepted by armijo condition.")
                set_ftype!(solver, "F")
                set_alpha!(solver, alpha_soc)
                return true
            end
        else
            # Case II
            if is_sufficient_progress(theta_soc,theta,get_opt(solver).gamma_theta,varphi_soc,varphi,get_opt(solver).gamma_phi,has_constraints(solver))
                @trace(get_logger(solver),"Step in second order correction accepted by sufficient progress.")
                set_ftype!(solver, "H")
                set_alpha!(solver, alpha_soc)
                return true
            end
        end

        theta_soc>get_opt(solver).kappa_soc*theta_soc_old && break
        theta_soc_old = theta_soc
    end
    @trace(get_logger(solver),"Second-order correction terminated.")

    return false
end


function inertia_correction!(
    inertia_corrector::InertiaBased,
    solver::AbstractMadNLPSolver{T}
) where {T}

    n_trial = 0
    del_w_prev = zero(T)
    del_c_prev = zero(T)
    set_del_w!(solver, zero(T))
    set_del_c!(solver, zero(T))

    @trace(get_logger(solver),"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    num_pos,num_zero,num_neg = inertia(get_kkt(solver).linear_solver)


    solve_status = if is_inertia_correct(get_kkt(solver), num_pos, num_zero, num_neg)
        # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
        solve_refine_wrapper!(get_d(solver), solver, get_p(solver), get__w4(solver))
    else
        false
    end

    while !solve_status
        @debug(get_logger(solver),"Primal-dual perturbed.")

        if n_trial == 0
            set_del_w!(solver, get_del_w_last(solver)==zero(T) ? get_opt(solver).first_hessian_perturbation :
                max(get_opt(solver).min_hessian_perturbation,get_opt(solver).perturb_dec_fact*get_del_w_last(solver))
                    )
        else
            set_del_w!(solver, get_del_w(solver) * (get_del_w_last(solver)==zero(T) ? get_opt(solver).perturb_inc_fact_first : get_opt(solver).perturb_inc_fact))
            if get_del_w(solver)>get_opt(solver).max_hessian_perturbation
                get_cnt(solver).k+=1
                @debug(get_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        set_del_c!(solver, (num_zero == 0 ? zero(T) : get_opt(solver).jacobian_regularization_value * get_mu(solver)^(get_opt(solver).jacobian_regularization_exponent)))
        regularize_diagonal!(get_kkt(solver), get_del_w(solver) - del_w_prev, get_del_c(solver) - del_c_prev)
        del_w_prev = get_del_w(solver)
        del_c_prev = get_del_c(solver)

        factorize_wrapper!(solver)
        num_pos,num_zero,num_neg = inertia(get_kkt(solver).linear_solver)

        solve_status = if is_inertia_correct(get_kkt(solver), num_pos, num_zero, num_neg)
            solve_refine_wrapper!(get_d(solver), solver, get_p(solver), get__w4(solver))
        else
            false
        end

        n_trial += 1
    end

    get_del_w(solver) != 0 && (set_del_w_last!(solver, get_del_w(solver)))
    return true
end

function inertia_correction!(
    inertia_corrector::InertiaFree,
    solver::AbstractMadNLPSolver{T}
    ) where T
    n_trial = 0
    del_w_prev = zero(T)
    del_c_prev = zero(T)
    set_del_w!(solver, zero(T))
    set_del_c!(solver, zero(T))

    @trace(get_logger(solver),"Inertia-free regularization started.")
    dx = primal(get_d(solver))
    p0 = inertia_corrector.p0
    d0 = inertia_corrector.d0
    t = inertia_corrector.t
    n = primal(d0)
    wx= inertia_corrector.wx
    g = inertia_corrector.g

    set_g_ifr!(solver,g)
    set_aug_rhs_ifr!(solver, get_kkt(solver), p0)

    factorize_wrapper!(solver)

    solve_status = solve_refine_wrapper!(
        d0, solver, p0, get__w3(solver),
    ) && solve_refine_wrapper!(
        get_d(solver), solver, get_p(solver), get__w4(solver),
    )
    copyto!(t,dx)
    axpy!(-1.,n,t)

    while !curv_test(t,n,g,get_kkt(solver),wx,get_opt(solver).inertia_free_tol) || !solve_status
        @debug(get_logger(solver),"Primal-dual perturbed.")
        if n_trial == 0
            set_del_w!(solver, get_del_w_last(solver)==.0 ? get_opt(solver).first_hessian_perturbation :
                max(get_opt(solver).min_hessian_perturbation,get_opt(solver).perturb_dec_fact*get_del_w_last(solver))
                    )
        else
            set_del_w!(solver, get_del_w(solver) * (get_del_w_last(solver)==.0 ? get_opt(solver).perturb_inc_fact_first : get_opt(solver).perturb_inc_fact))
            if get_del_w(solver)>get_opt(solver).max_hessian_perturbation
                get_cnt(solver).k+=1
                @debug(get_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        set_del_c!(solver, get_opt(solver).jacobian_regularization_value * get_mu(solver)^(get_opt(solver).jacobian_regularization_exponent))
        regularize_diagonal!(get_kkt(solver), get_del_w(solver) - del_w_prev, get_del_c(solver) - del_c_prev)
        del_w_prev = get_del_w(solver)
        del_c_prev = get_del_c(solver)

        factorize_wrapper!(solver)
        solve_status = solve_refine_wrapper!(
            d0, solver, p0, get__w3(solver)
        ) && solve_refine_wrapper!(
            get_d(solver), solver, get_p(solver), get__w4(solver)
        )
        copyto!(t,dx)
        axpy!(-1.,n,t)

        n_trial += 1
    end

    get_del_w(solver) != 0 && (set_del_w_last!(solver, get_del_w(solver)))
    return true
end

function inertia_correction!(
    inertia_corrector::InertiaIgnore,
    solver::AbstractMadNLPSolver{T}
    ) where T

    n_trial = 0
    del_w_prev = zero(T)
    del_c_prev = zero(T)
    set_del_w!(solver, zero(T))
    set_del_c!(solver, zero(T))

    @trace(get_logger(solver),"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    solve_status = solve_refine_wrapper!(
        get_d(solver), solver, get_p(solver), get__w4(solver),
    )
    while !solve_status
        @debug(get_logger(solver),"Primal-dual perturbed.")
        if n_trial == 0
            set_del_w!(solver, get_del_w_last(solver)==zero(T) ? get_opt(solver).first_hessian_perturbation :
                max(get_opt(solver).min_hessian_perturbation,get_opt(solver).perturb_dec_fact*get_del_w_last(solver)))
        else
            set_del_w!(solver, get_del_w(solver) * (get_del_w_last(solver)==zero(T) ? get_opt(solver).perturb_inc_fact_first : get_opt(solver).perturb_inc_fact))
            if get_del_w(solver)>get_opt(solver).max_hessian_perturbation
                get_cnt(solver).k+=1
                @debug(get_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        set_del_c!(solver, get_opt(solver).jacobian_regularization_value * get_mu(solver)^(get_opt(solver).jacobian_regularization_exponent))
        regularize_diagonal!(get_kkt(solver), get_del_w(solver) - del_w_prev, get_del_c(solver) - del_c_prev)
        del_w_prev = get_del_w(solver)
        del_c_prev = get_del_c(solver)

        factorize_wrapper!(solver)
        solve_status = solve_refine_wrapper!(
            get_d(solver), solver, get_p(solver), get__w4(solver)
        )
        n_trial += 1
    end
    get_del_w(solver) != 0 && (set_del_w_last!(solver, get_del_w(solver)))
    return true
end

function curv_test(t,n,g,kkt,wx,inertia_free_tol)
    mul_hess_blk!(wx, kkt, t)
    dot(wx,t) + max(dot(wx,n)-dot(g,n),0) - inertia_free_tol*dot(t,t) >=0
end
