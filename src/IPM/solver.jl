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
    _nlp(solver), solver;
    kwargs...)


function initialize!(solver::AbstractMadNLPSolver{T}) where T

    nlp = _nlp(solver)
    opt = _opt(solver)

    # Initializing variables
    @trace(_logger(solver),"Initializing variables.")
    initialize!(
        _cb(solver),
        _x(solver),
        _xl(solver),
        _xu(solver),
        _y(solver),
        _rhs(solver),
        _ind_ineq(solver);
        tol=opt.bound_relax_factor,
        bound_push=opt.bound_push,
        bound_fac=opt.bound_fac,
    )
    fill!(_jacl(solver), zero(T))
    fill!(_zl_r(solver), one(T))
    fill!(_zu_r(solver), one(T))

    # Initializing scaling factors
    if opt.nlp_scaling
        set_scaling!(
            _cb(solver),
            _x(solver),
            _xl(solver),
            _xu(solver),
            _y(solver),
            _rhs(solver),
            _ind_ineq(solver),
            opt.nlp_scaling_max_gradient
        )
    end

    # Initializing KKT system
    initialize!(_kkt(solver))

    # Initializing jacobian and gradient
    eval_jac_wrapper!(solver, _kkt(solver), _x(solver))
    eval_grad_f_wrapper!(solver, _f(solver),_x(solver))


    @trace(_logger(solver),"Initializing constraint duals.")
    if !_opt(solver).dual_initialized
        initialize_dual(solver, opt.dual_initialization_method)
    end

    # Initializing
    set_obj_val!(solver, eval_f_wrapper(solver, _x(solver)))
    eval_cons_wrapper!(solver, _c(solver), _x(solver))
    eval_lag_hess_wrapper!(solver, _kkt(solver), _x(solver), _y(solver))

    theta = get_theta(_c(solver))
    set_theta_max!(solver, T(1e4)*max(one(T),theta))
    set_theta_min!(solver, T(1e-4)*max(one(T),theta))
    set_mu!(solver, _opt(solver).mu_init)
    set_tau!(solver, max(_opt(solver).tau_min,one(T)-_opt(solver).mu_init))
    push!(_filter(solver), (_theta_max(solver),-T(Inf)))

    return REGULAR
end

abstract type DualInitializeOptions end
struct DualInitializeSetZero <: DualInitializeOptions end
struct DualInitializeLeastSquares <: DualInitializeOptions end

function initialize_dual(solver::AbstractMadNLPSolver{T}, ::Type{DualInitializeSetZero}) where T
    fill!(_y(solver), zero(T))
end
function initialize_dual(solver::AbstractMadNLPSolver{T}, ::Type{DualInitializeLeastSquares}) where T
    set_initial_rhs!(solver, _kkt(solver))
    factorize_wrapper!(solver)
    is_solved = solve_refine_wrapper!(
        _d(solver), solver, _p(solver), __w4(solver)
    )
    if !is_solved || (norm(dual(_d(solver)), T(Inf)) > _opt(solver).constr_mult_init_max)
        fill!(_y(solver), zero(T))
    else
        copyto!(_y(solver), dual(_d(solver)))
    end
end

function reinitialize!(solver::AbstractMadNLPSolver)
    variable(_x(solver)) .= get_x0(_nlp(solver))

    set_obj_val!(solver, eval_f_wrapper(solver, _x(solver)))
    eval_grad_f_wrapper!(solver, _f(solver), _x(solver))
    eval_cons_wrapper!(solver, _c(solver), _x(solver))
    eval_jac_wrapper!(solver, _kkt(solver), _x(solver))
    eval_lag_hess_wrapper!(solver, _kkt(solver), _x(solver), _y(solver))

    theta = get_theta(_c(solver))
    set_theta_max!(solver, 1e4*max(1,theta))
    set_theta_min!(solver, 1e-4*max(1,theta))
    set_mu!(solver, _opt(solver).mu_init)
    set_tau!(solver, max(_opt(solver).tau_min,1-_opt(solver).mu_init))
    empty!(_filter(solver))
    push!(_filter(solver), (_theta_max(solver),-Inf))

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
        full(_x(solver))[1:get_nvar(nlp)] .= x
    end
    if y != nothing
        _y(solver)[1:get_ncon(nlp)] .= y
    end
    if zl != nothing
        full(_zl(solver))[1:get_nvar(nlp)] .= zl
    end
    if zu != nothing
        full(_zu(solver))[1:get_nvar(nlp)] .= zu
    end

    if !isempty(kwargs)
        @warn(_logger(solver),"The options set during resolve may not have an effect")
        set_options!(_opt(solver), kwargs)
    end

    try
        if _status(solver) == INITIAL
            @notice(_logger(solver),"This is $(introduce()), running with $(introduce(_kkt(solver).linear_solver))\n")
            print_init(solver)
            set_status!(solver, initialize!(solver))
        else # resolving the problem
            set_status!(solver, reinitialize!(solver))
        end

        while _status(solver) >= REGULAR
            _status(solver) == REGULAR && (set_status!(solver, regular!(solver)))
            _status(solver) == RESTORE && (set_status!(solver, restore!(solver)))
            _status(solver) == ROBUST && (set_status!(solver, robust!(solver)))
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
            _opt(solver).rethrow_error && rethrow(e)
        elseif e isa InterruptException
            set_status!(solver, USER_REQUESTED_STOP)
            _opt(solver).rethrow_error && rethrow(e)
        else
            set_status!(solver, INTERNAL_ERROR)
            _opt(solver).rethrow_error && rethrow(e)
        end
    finally
        _cnt(solver).total_time = time() - _cnt(solver).start_time
        if !(_status(solver) < SOLVE_SUCCEEDED)
            print_summary(solver)
        end
        @notice(_logger(solver),"$(Base.text_colors[color_status(_status(solver))])EXIT: $(get_status_output(_status(solver), _opt(solver)))$(Base.text_colors[:normal])")
        _opt(solver).disable_garbage_collector &&
            (GC.enable(true); @warn(_logger(solver),"Julia garbage collector is turned back on"))
        finalize(_logger(solver))

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
        (status = evaluate_termination_criteria!(solver::AbstractMadNLPSolver)) == REGULAR || return status

        # update the barrier parameter
        update_mu!(solver)

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
    # Backup previous point and zero out del_w and alpha_z
    initialize_restore!(solver)

    F = get_F(
        _c(solver),
        primal(_f(solver)),
        primal(_zl(solver)),
        primal(_zu(solver)),
        _jacl(solver),
        _x_lr(solver),
        _xl_r(solver),
        _zl_r(solver),
        _xu_r(solver),
        _x_ur(solver),
        _zu_r(solver),
        _mu(solver),
    )

    while true
        take_ftb_step!(solver)

        evaluate_for_sufficient_decrease_restore!(solver)

        F_trial = get_F(
            _c(solver),
            primal(_f(solver)),
            primal(_zl(solver)),
            primal(_zu(solver)),
            _jacl(solver),
            _x_lr(solver),
            _xl_r(solver),
            _zl_r(solver),
            _xu_r(solver),
            _x_ur(solver),
            _zu_r(solver),
            _mu(solver),
        )
        # Check for sufficient decrease
        if F_trial > _opt(solver).soft_resto_pderror_reduction_factor*F
            # Sufficient decrease not observed, backing up and starting robust restorer.
            backtrack_restore!(solver)
            return ROBUST
        end
        F = F_trial

        adjust_boundary!(_x_lr(solver),_xl_r(solver),_x_ur(solver),_xu_r(solver),_mu(solver))

        (status = evaluate_termination_criteria_restore!(solver)) == RESTORE || return status

        evaluate_for_next_iter_restore!(solver)

        print_iter(solver)

        compute_newton_step_restore!(solver)

        set_ftype!(solver, "f")
    end
end

function robust!(solver::AbstractMadNLPSolver{T}) where T
    initialize_robust_restorer!(solver)
    RR = _RR(solver)
    while true
        # evaluate termination criteria
        eval_for_next_iter_RR!(solver)

        print_iter(solver;is_resto=true)

        (status = evaluate_termination_criteria_RR!(solver)) == ROBUST || return status

        # update the barrier parameter
        update_mu_RR!(solver)

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
    @trace(_logger(solver),"Second-order correction started.")

    wx = primal(__w1(solver))
    wy = dual(__w1(solver))
    copyto!(wy, _c_trial(solver))
    axpy!(alpha_max, _c(solver), wy)

    theta_soc_old = theta_trial
    for p=1:_opt(solver).max_soc
        # compute second order correction
        set_aug_rhs!(solver, _kkt(solver), wy)
        dual_inf_perturbation!(
            primal(_p(solver)),
            _ind_llb(solver),_ind_uub(solver),_mu(solver),_opt(solver).kappa_d,
        )
        solve_refine_wrapper!(
            __w1(solver), solver, _p(solver), __w4(solver)
        )
        alpha_soc = get_alpha_max(
            primal(_x(solver)),
            primal(_xl(solver)),
            primal(_xu(solver)),
            wx,_tau(solver)
        )

        copyto!(primal(_x_trial(solver)), primal(_x(solver)))
        axpy!(alpha_soc, wx, primal(_x_trial(solver)))
        eval_cons_wrapper!(solver, _c_trial(solver), _x_trial(solver))
        set_obj_val_trial!(solver, eval_f_wrapper(solver, _x_trial(solver)))

        theta_soc = get_theta(_c_trial(solver))
        varphi_soc= get_varphi(_obj_val_trial(solver),_x_trial_lr(solver),_xl_r(solver),_xu_r(solver),_x_trial_ur(solver),_mu(solver))

        !is_filter_acceptable(_filter(solver),theta_soc,varphi_soc) && break

        if theta <=_theta_min(solver) && switching_condition
            # Case I
            if is_armijo(varphi_soc,varphi,_opt(solver).eta_phi,_alpha(solver),varphi_d)
                @trace(_logger(solver),"Step in second order correction accepted by armijo condition.")
                set_ftype!(solver, "F")
                set_alpha!(solver, alpha_soc)
                return true
            end
        else
            # Case II
            if is_sufficient_progress(theta_soc,theta,_opt(solver).gamma_theta,varphi_soc,varphi,_opt(solver).gamma_phi,has_constraints(solver))
                @trace(_logger(solver),"Step in second order correction accepted by sufficient progress.")
                set_ftype!(solver, "H")
                set_alpha!(solver, alpha_soc)
                return true
            end
        end

        theta_soc>_opt(solver).kappa_soc*theta_soc_old && break
        theta_soc_old = theta_soc
    end
    @trace(_logger(solver),"Second-order correction terminated.")

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

    @trace(_logger(solver),"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    num_pos,num_zero,num_neg = inertia(_kkt(solver).linear_solver)


    solve_status = !is_inertia_correct(_kkt(solver), num_pos, num_zero, num_neg) ?
        false : solve_refine_wrapper!(
            _d(solver), solver, _p(solver), __w4(solver),
        )


    while !solve_status
        @debug(_logger(solver),"Primal-dual perturbed.")

        if n_trial == 0
            set_del_w!(solver, _del_w_last(solver)==zero(T) ? _opt(solver).first_hessian_perturbation :
                max(_opt(solver).min_hessian_perturbation,_opt(solver).perturb_dec_fact*_del_w_last(solver))
                    )
        else
            set_del_w!(solver, _del_w(solver) * (_del_w_last(solver)==zero(T) ? _opt(solver).perturb_inc_fact_first : _opt(solver).perturb_inc_fact))
            if _del_w(solver)>_opt(solver).max_hessian_perturbation
                _cnt(solver).k+=1
                @debug(_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        set_del_c!(solver, (num_zero == 0 ? zero(T) : _opt(solver).jacobian_regularization_value * _mu(solver)^(_opt(solver).jacobian_regularization_exponent)))
        regularize_diagonal!(_kkt(solver), _del_w(solver) - del_w_prev, _del_c(solver) - del_c_prev)
        del_w_prev = _del_w(solver)
        del_c_prev = _del_c(solver)

        factorize_wrapper!(solver)
        num_pos,num_zero,num_neg = inertia(_kkt(solver).linear_solver)

        solve_status = !is_inertia_correct(_kkt(solver), num_pos, num_zero, num_neg) ?
            false : solve_refine_wrapper!(
                _d(solver), solver, _p(solver), __w4(solver)
            )
        n_trial += 1
    end

    _del_w(solver) != 0 && (set_del_w_last!(solver, _del_w(solver)))
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

    @trace(_logger(solver),"Inertia-free regularization started.")
    dx = primal(_d(solver))
    p0 = inertia_corrector.p0
    d0 = inertia_corrector.d0
    t = inertia_corrector.t
    n = primal(d0)
    wx= inertia_corrector.wx
    g = inertia_corrector.g

    set_g_ifr!(solver,g)
    set_aug_rhs_ifr!(solver, _kkt(solver), p0)

    factorize_wrapper!(solver)

    solve_status = solve_refine_wrapper!(
        d0, solver, p0, __w3(solver),
    ) && solve_refine_wrapper!(
        _d(solver), solver, _p(solver), __w4(solver),
    )
    copyto!(t,dx)
    axpy!(-1.,n,t)

    while !curv_test(t,n,g,_kkt(solver),wx,_opt(solver).inertia_free_tol) || !solve_status
        @debug(_logger(solver),"Primal-dual perturbed.")
        if n_trial == 0
            set_del_w!(solver, _del_w_last(solver)==.0 ? _opt(solver).first_hessian_perturbation :
                max(_opt(solver).min_hessian_perturbation,_opt(solver).perturb_dec_fact*_del_w_last(solver))
                    )
        else
            set_del_w!(solver, _del_w(solver) * (_del_w_last(solver)==.0 ? _opt(solver).perturb_inc_fact_first : _opt(solver).perturb_inc_fact))
            if _del_w(solver)>_opt(solver).max_hessian_perturbation
                _cnt(solver).k+=1
                @debug(_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        set_del_c!(solver, _opt(solver).jacobian_regularization_value * _mu(solver)^(_opt(solver).jacobian_regularization_exponent))
        regularize_diagonal!(_kkt(solver), _del_w(solver) - del_w_prev, _del_c(solver) - del_c_prev)
        del_w_prev = _del_w(solver)
        del_c_prev = _del_c(solver)

        factorize_wrapper!(solver)
        solve_status = solve_refine_wrapper!(
            d0, solver, p0, __w3(solver)
        ) && solve_refine_wrapper!(
            _d(solver), solver, _p(solver), __w4(solver)
        )
        copyto!(t,dx)
        axpy!(-1.,n,t)

        n_trial += 1
    end

    _del_w(solver) != 0 && (set_del_w_last!(solver, _del_w(solver)))
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

    @trace(_logger(solver),"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    solve_status = solve_refine_wrapper!(
        _d(solver), solver, _p(solver), __w4(solver),
    )
    while !solve_status
        @debug(_logger(solver),"Primal-dual perturbed.")
        if n_trial == 0
            set_del_w!(solver, _del_w_last(solver)==zero(T) ? _opt(solver).first_hessian_perturbation :
                max(_opt(solver).min_hessian_perturbation,_opt(solver).perturb_dec_fact*_del_w_last(solver)))
        else
            set_del_w!(solver, _del_w(solver) * (_del_w_last(solver)==zero(T) ? _opt(solver).perturb_inc_fact_first : _opt(solver).perturb_inc_fact))
            if _del_w(solver)>_opt(solver).max_hessian_perturbation
                _cnt(solver).k+=1
                @debug(_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        set_del_c!(solver, _opt(solver).jacobian_regularization_value * _mu(solver)^(_opt(solver).jacobian_regularization_exponent))
        regularize_diagonal!(_kkt(solver), _del_w(solver) - del_w_prev, _del_c(solver) - del_c_prev)
        del_w_prev = _del_w(solver)
        del_c_prev = _del_c(solver)

        factorize_wrapper!(solver)
        solve_status = solve_refine_wrapper!(
            _d(solver), solver, _p(solver), __w4(solver)
        )
        n_trial += 1
    end
    _del_w(solver) != 0 && (set_del_w_last!(solver, _del_w(solver)))
    return true
end

function curv_test(t,n,g,kkt,wx,inertia_free_tol)
    mul_hess_blk!(wx, kkt, t)
    dot(wx,t) + max(dot(wx,n)-dot(g,n),0) - inertia_free_tol*dot(t,t) >=0
end
