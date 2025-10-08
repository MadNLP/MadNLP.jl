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
    _obj_val!(solver, eval_f_wrapper(solver, _x(solver)))
    eval_cons_wrapper!(solver, _c(solver), _x(solver))
    eval_lag_hess_wrapper!(solver, _kkt(solver), _x(solver), _y(solver))

    theta = get_theta(_c(solver))
    _theta_max!(solver, 1e4*max(1,theta))
    _theta_min!(solver, 1e-4*max(1,theta))
    _mu!(solver, _opt(solver).mu_init)
    _tau!(solver, max(_opt(solver).tau_min,1-_opt(solver).mu_init))
    push!(_filter(solver), (_theta_max(solver),-Inf))

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
    if !is_solved || (norm(dual(_d(solver)), Inf) > _opt(solver).constr_mult_init_max)
        fill!(_y(solver), zero(T))
    else
        copyto!(_y(solver), dual(_d(solver)))
    end
end

function reinitialize!(solver::AbstractMadNLPSolver)
    variable(_x(solver)) .= get_x0(_nlp(solver))

    _obj_val!(solver, eval_f_wrapper(solver, _x(solver)))
    eval_grad_f_wrapper!(solver, _f(solver), _x(solver))
    eval_cons_wrapper!(solver, _c(solver), _x(solver))
    eval_jac_wrapper!(solver, _kkt(solver), _x(solver))
    eval_lag_hess_wrapper!(solver, _kkt(solver), _x(solver), _y(solver))

    theta = get_theta(_c(solver))
    _theta_max!(solver, 1e4*max(1,theta))
    _theta_min!(solver, 1e-4*max(1,theta))
    _mu!(solver, _opt(solver).mu_init)
    _tau!(solver, max(_opt(solver).tau_min,1-_opt(solver).mu_init))
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
            _status!(solver, initialize!(solver))
        else # resolving the problem
            _status!(solver, reinitialize!(solver))
        end

        while _status(solver) >= REGULAR
            _status(solver) == REGULAR && (_status!(solver, regular!(solver)))
            _status(solver) == RESTORE && (_status!(solver, restore!(solver)))
            _status(solver) == ROBUST && (_status!(solver, robust!(solver)))
        end
    catch e
        if e isa InvalidNumberException
            if e.callback == :obj
                _status!(solver, INVALID_NUMBER_OBJECTIVE)
            elseif e.callback == :grad
                _status!(solver, INVALID_NUMBER_GRADIENT)
            elseif e.callback == :cons
                _status!(solver, INVALID_NUMBER_CONSTRAINTS)
            elseif e.callback == :jac
                _status!(solver, INVALID_NUMBER_JACOBIAN)
            elseif e.callback == :hess
                _status!(solver, INVALID_NUMBER_HESSIAN_LAGRANGIAN)
            else
                _status!(solver, INVALID_NUMBER_DETECTED)
            end
        elseif e isa NotEnoughDegreesOfFreedomException
            _status!(solver, NOT_ENOUGH_DEGREES_OF_FREEDOM)
        elseif e isa LinearSolverException
            _status!(solver, ERROR_IN_STEP_COMPUTATION;)
            _opt(solver).rethrow_error && rethrow(e)
        elseif e isa InterruptException
            _status!(solver, USER_REQUESTED_STOP)
            _opt(solver).rethrow_error && rethrow(e)
        else
            _status!(solver, INTERNAL_ERROR)
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
        if (_cnt(solver).k!=0 && !_opt(solver).jacobian_constant)
            eval_jac_wrapper!(solver, _kkt(solver), _x(solver))
        end

        jtprod!(_jacl(solver), _kkt(solver), _y(solver))
        sd = get_sd(_y(solver),_zl_r(solver),_zu_r(solver),T(_opt(solver).s_max))
        sc = get_sc(_zl_r(solver),_zu_r(solver),T(_opt(solver).s_max))
        _inf_pr!(solver, get_inf_pr(_c(solver)))
        _inf_du!(solver, get_inf_du(
            full(_f(solver)),
            full(_zl(solver)),
            full(_zu(solver)),
            _jacl(solver),
            sd,
        ))
        _inf_compl!(solver, _inf_compl(solver, sc; mu=zero(T)))
        inf_compl_mu = _inf_compl(solver, sc)

        print_iter(solver)

        # evaluate termination criteria
        @trace(_logger(solver),"Evaluating termination criteria.")
        _inf_total(solver) <= _opt(solver).tol && return SOLVE_SUCCEEDED
        _inf_total(solver) <= _opt(solver).acceptable_tol ?
            (_cnt(solver).acceptable_cnt < _opt(solver).acceptable_iter ?
            _cnt(solver).acceptable_cnt+=1 : return SOLVED_TO_ACCEPTABLE_LEVEL) : (_cnt(solver).acceptable_cnt = 0)
        _inf_total(solver) >= _opt(solver).diverging_iterates_tol && return DIVERGING_ITERATES
        _cnt(solver).k>=_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-_cnt(solver).start_time>=_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        @trace(_logger(solver),"Updating the barrier parameter.")
        while _mu(solver) != max(_opt(solver).mu_min,_opt(solver).tol/10) &&
            max(_inf_pr(solver),_inf_du(solver),inf_compl_mu) <= _opt(solver).barrier_tol_factor*_mu(solver)
            mu_new = get_mu(_mu(solver),_opt(solver).mu_min,
                            _opt(solver).mu_linear_decrease_factor,_opt(solver).mu_superlinear_decrease_power,_opt(solver).tol)
            inf_compl_mu = get_inf_compl(_x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),_mu(solver),sc)
            _tau!(solver, get_tau(_mu(solver),_opt(solver).tau_min))
            _mu!(solver, mu_new)
            empty!(_filter(solver))
            push!(_filter(solver),(_theta_max(solver),-Inf))
        end

        # compute the newton step
        @trace(_logger(solver),"Computing the newton step.")
        if (_cnt(solver).k!=0 && !_opt(solver).hessian_constant)
            eval_lag_hess_wrapper!(solver, _kkt(solver), _x(solver), _y(solver))
        end

        set_aug_diagonal!(_kkt(solver),solver)
        set_aug_rhs!(solver, _kkt(solver), _c(solver))
        dual_inf_perturbation!(primal(_p(solver)),_ind_llb(solver),_ind_uub(solver),_mu(solver),_opt(solver).kappa_d)

        inertia_correction!(_inertia_corrector(solver), solver) || return ROBUST

        @trace(_logger(solver),"Backtracking line search initiated.")
        status = filter_line_search!(solver)
        if status != LINESEARCH_SUCCEEDED
            return status
        end

        @trace(_logger(solver),"Updating primal-dual variables.")
        copyto!(full(_x(solver)), full(_x_trial(solver)))
        copyto!(_c(solver), _c_trial(solver))
        _obj_val!(solver, _obj_val_trial(solver))
        adjust_boundary!(_x_lr(solver),_xl_r(solver),_x_ur(solver),_xu_r(solver),_mu(solver))

        axpy!(_alpha(solver),dual(_d(solver)),_y(solver))

        _zl_r(solver) .+= _alpha_z(solver) .* dual_lb(_d(solver))
        _zu_r(solver) .+= _alpha_z(solver) .* dual_ub(_d(solver))
        reset_bound_dual!(
            primal(_zl(solver)),
            primal(_x(solver)),
            primal(_xl(solver)),
            _mu(solver),_opt(solver).kappa_sigma,
        )
        reset_bound_dual!(
            primal(_zu(solver)),
            primal(_xu(solver)),
            primal(_x(solver)),
            _mu(solver),_opt(solver).kappa_sigma,
        )

        eval_grad_f_wrapper!(solver, _f(solver),_x(solver))

        _cnt(solver).k+=1
            @trace(_logger(solver),"Proceeding to the next interior point iteration.")
    end
end


function restore!(solver::AbstractMadNLPSolver{T}) where T
    _del_w!(solver, 0)
    # Backup the previous primal iterate
    copyto!(primal(__w1(solver)), full(_x(solver)))
    copyto!(dual(__w1(solver)), _y(solver))
    copyto!(dual(__w2(solver)), _c(solver))

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
    _alpha_z!(solver, zero(T))
    _ftype!(solver, "R")
    while true
        alpha_max = get_alpha_max(
            primal(_x(solver)),
            primal(_xl(solver)),
            primal(_xu(solver)),
            primal(_d(solver)),
            _tau(solver),
        )
        _alpha!(solver,min(
            alpha_max,
            get_alpha_z(_zl_r(solver),_zu_r(solver),dual_lb(_d(solver)),dual_ub(_d(solver)),_tau(solver)),
        ))

        axpy!(_alpha(solver), primal(_d(solver)), full(_x(solver)))
        axpy!(_alpha(solver), dual(_d(solver)), _y(solver))
        _zl_r(solver) .+= _alpha(solver) .* dual_lb(_d(solver))
        _zu_r(solver) .+= _alpha(solver) .* dual_ub(_d(solver))

        eval_cons_wrapper!(solver,_c(solver),_x(solver))
        eval_grad_f_wrapper!(solver,_f(solver),_x(solver))
        _obj_val!(solver, eval_f_wrapper(solver,_x(solver)))

        !_opt(solver).jacobian_constant && eval_jac_wrapper!(solver,_kkt(solver),_x(solver))
        jtprod!(_jacl(solver),_kkt(solver),_y(solver))

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
        if F_trial > _opt(solver).soft_resto_pderror_reduction_factor*F
            copyto!(primal(_x(solver)), primal(__w1(solver)))
            copyto!(_y(solver), dual(__w1(solver)))
            copyto!(_c(solver), dual(__w2(solver))) # backup the previous primal iterate
            return ROBUST
        end

        adjust_boundary!(_x_lr(solver),_xl_r(solver),_x_ur(solver),_xu_r(solver),_mu(solver))

        F = F_trial

        theta = get_theta(_c(solver))
        varphi= get_varphi(_obj_val(solver),_x_lr(solver),_xl_r(solver),_xu_r(solver),_x_ur(solver),_mu(solver))

        _cnt(solver).k+=1

        is_filter_acceptable(_filter(solver),theta,varphi) ? (return REGULAR) : (_cnt(solver).t+=1)
        _cnt(solver).k>=_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-_cnt(solver).start_time>=_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED


        sd = get_sd(_y(solver),_zl_r(solver),_zu_r(solver),_opt(solver).s_max)
        sc = get_sc(_zl_r(solver),_zu_r(solver),_opt(solver).s_max)
        _inf_pr!(solver, get_inf_pr(_c(solver)))
        _inf_du!(solver, get_inf_du(
            primal(_f(solver)),
            primal(_zl(solver)),
            primal(_zu(solver)),
            _jacl(solver),
            sd,
        ))

        _inf_compl!(solver, get_inf_compl(_x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),zero(T),sc))
        inf_compl_mu = get_inf_compl(_x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),_mu(solver),sc)
        print_iter(solver)

        !_opt(solver).hessian_constant && eval_lag_hess_wrapper!(solver,_kkt(solver),_x(solver),_y(solver))
        set_aug_diagonal!(_kkt(solver),solver)
        set_aug_rhs!(solver, _kkt(solver), _c(solver))

        dual_inf_perturbation!(primal(_p(solver)),_ind_llb(solver),_ind_uub(solver),_mu(solver),_opt(solver).kappa_d)
        factorize_wrapper!(solver)
        solve_refine_wrapper!(
            _d(solver), solver, _p(solver), __w4(solver)
        )

        _ftype!(solver, "f")
    end
end

function robust!(solver::AbstractMadNLPSolver{T}) where T
    initialize_robust_restorer!(solver)
    RR = _RR(solver)
    while true
        if !_opt(solver).jacobian_constant
            eval_jac_wrapper!(solver, _kkt(solver), _x(solver))
        end
        jtprod!(_jacl(solver), _kkt(solver), _y(solver))

        # evaluate termination criteria
        @trace(_logger(solver),"Evaluating restoration phase termination criteria.")
        sd = get_sd(_y(solver),_zl_r(solver),_zu_r(solver),_opt(solver).s_max)
        sc = get_sc(_zl_r(solver),_zu_r(solver),_opt(solver).s_max)
        _inf_pr!(solver, get_inf_pr(_c(solver)))
        _inf_du!(get_inf_du(
            primal(_f(solver)),
            primal(_zl(solver)),
            primal(_zu(solver)),
            _jacl(solver),
            sd,
        ))
        _inf_compl!(solver, get_inf_compl(_x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),zero(T),sc))

        # Robust restoration phase error
        RR.inf_pr_R = get_inf_pr_R(_c(solver),RR.pp,RR.nn)
        RR.inf_du_R = get_inf_du_R(RR.f_R,_y(solver),primal(_zl(solver)),primal(_zu(solver)),_jacl(solver),RR.zp,RR.zn,_opt(solver).rho,sd)
        RR.inf_compl_R = get_inf_compl_R(
            _x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),RR.pp,RR.zp,RR.nn,RR.zn,zero(T),sc)
        inf_compl_mu_R = get_inf_compl_R(
            _x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,sc)

        print_iter(solver;is_resto=true)

        max(RR.inf_pr_R,RR.inf_du_R,RR.inf_compl_R) <= _opt(solver).tol && return INFEASIBLE_PROBLEM_DETECTED
        _cnt(solver).k>=_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-_cnt(solver).start_time>=_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        @trace(_logger(solver),"Updating restoration phase barrier parameter.")
        while RR.mu_R >= _opt(solver).mu_min &&
            max(RR.inf_pr_R,RR.inf_du_R,inf_compl_mu_R) <= _opt(solver).barrier_tol_factor*RR.mu_R
            RR.mu_R = get_mu(RR.mu_R,_opt(solver).mu_min,
                             _opt(solver).mu_linear_decrease_factor,_opt(solver).mu_superlinear_decrease_power,_opt(solver).tol)
            inf_compl_mu_R = get_inf_compl_R(
                _x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver),RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,sc)
            RR.tau_R= max(_opt(solver).tau_min,1-RR.mu_R)
            RR.zeta = sqrt(RR.mu_R)

            empty!(RR.filter)
            push!(RR.filter,(_theta_max(solver),-Inf))
        end

        # compute the newton step
        if !_opt(solver).hessian_constant
            eval_lag_hess_wrapper!(solver, _kkt(solver), _x(solver), _y(solver); is_resto=true)
        end
        set_aug_RR!(_kkt(solver), solver, RR)

        # without inertia correction,
        @trace(_logger(solver),"Solving restoration phase primal-dual system.")
        set_aug_rhs_RR!(solver, _kkt(solver), RR, _opt(solver).rho)

        inertia_correction!(_inertia_corrector(solver), solver) || return RESTORATION_FAILED


        finish_aug_solve_RR!(
            RR.dpp,RR.dnn,RR.dzp,RR.dzn,_y(solver),dual(_d(solver)),
            RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,_opt(solver).rho
        )

        # filter start
        @trace(_logger(solver),"Backtracking line search initiated.")
        status = filter_line_search_RR!(solver)
        if status != LINESEARCH_SUCCEEDED
            return status
        end

        @trace(_logger(solver),"Updating primal-dual variables.")
        copyto!(full(_x(solver)), full(_x_trial(solver)))
        copyto!(_c(solver), _c_trial(solver))
        copyto!(RR.pp, RR.pp_trial)
        copyto!(RR.nn, RR.nn_trial)

        RR.obj_val_R=RR.obj_val_R_trial
        set_f_RR!(solver,RR)

        axpy!(_alpha(solver), dual(_d(solver)), _y(solver))
        axpy!(_alpha_z(solver), RR.dzp,RR.zp)
        axpy!(_alpha_z(solver), RR.dzn,RR.zn)

        _zl_r(solver) .+= _alpha_z(solver) .* dual_lb(_d(solver))
        _zu_r(solver) .+= _alpha_z(solver) .* dual_ub(_d(solver))

        reset_bound_dual!(
            primal(_zl(solver)),
            primal(_x(solver)),
            primal(_xl(solver)),
            RR.mu_R, _opt(solver).kappa_sigma,
        )
        reset_bound_dual!(
            primal(_zu(solver)),
            primal(_xu(solver)),
            primal(_x(solver)),
            RR.mu_R, _opt(solver).kappa_sigma,
        )
        reset_bound_dual!(RR.zp,RR.pp,RR.mu_R,_opt(solver).kappa_sigma)
        reset_bound_dual!(RR.zn,RR.nn,RR.mu_R,_opt(solver).kappa_sigma)

        adjust_boundary!(_x_lr(solver),_xl_r(solver),_x_ur(solver),_xu_r(solver),_mu(solver))

        # check if going back to regular phase
        @trace(_logger(solver),"Checking if going back to regular phase.")
        _obj_val!(solver, eval_f_wrapper(solver, _x(solver)))
        eval_grad_f_wrapper!(solver, _f(solver), _x(solver))
        theta = get_theta(_c(solver))
        varphi= get_varphi(_obj_val(solver),_x_lr(solver),_xl_r(solver),_xu_r(solver),_x_ur(solver),_mu(solver))

        if is_filter_acceptable(_filter(solver),theta,varphi) &&
            theta <= _opt(solver).required_infeasibility_reduction * RR.theta_ref

            @trace(_logger(solver),"Going back to the regular phase.")
            set_initial_rhs!(solver, _kkt(solver))
            initialize!(_kkt(solver))

            factorize_wrapper!(solver)
            solve_refine_wrapper!(
                _d(solver), solver, _p(solver), __w4(solver)
            )
            if norm(dual(_d(solver)), Inf)>_opt(solver).constr_mult_init_max
                fill!(_y(solver), zero(T))
            else
                copyto!(_y(solver), dual(_d(solver)))
            end

            _cnt(solver).k+=1
            _cnt(solver).t+=1

            return REGULAR
        end

        _cnt(solver).k>=_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-_cnt(solver).start_time>=_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        @trace(_logger(solver),"Proceeding to the next restoration phase iteration.")
        _cnt(solver).k+=1
        _cnt(solver).t+=1
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
        _obj_val_trial!(solver, eval_f_wrapper(solver, _x_trial(solver)))

        theta_soc = get_theta(_c_trial(solver))
        varphi_soc= get_varphi(_obj_val_trial(solver),_x_trial_lr(solver),_xl_r(solver),_xu_r(solver),_x_trial_ur(solver),_mu(solver))

        !is_filter_acceptable(_filter(solver),theta_soc,varphi_soc) && break

        if theta <=_theta_min(solver) && switching_condition
            # Case I
            if is_armijo(varphi_soc,varphi,_opt(solver).eta_phi,_alpha(solver),varphi_d)
                @trace(_logger(solver),"Step in second order correction accepted by armijo condition.")
                _ftype!(solver, "F")
                _alpha!(solver, alpha_soc)
                return true
            end
        else
            # Case II
            if is_sufficient_progress(theta_soc,theta,_opt(solver).gamma_theta,varphi_soc,varphi,_opt(solver).gamma_phi,has_constraints(solver))
                @trace(_logger(solver),"Step in second order correction accepted by sufficient progress.")
                _ftype!(solver, "H")
                _alpha!(solver, alpha_soc)
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
    _del_w!(solver, zero(T))
    _del_c!(solver, zero(T))

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
            _del_w!(solver, _del_w_last(solver)==zero(T) ? _opt(solver).first_hessian_perturbation :
                max(_opt(solver).min_hessian_perturbation,_opt(solver).perturb_dec_fact*_del_w_last(solver))
                    )
        else
            _del_w!(solver, _del_w(solver) * (_del_w_last(solver)==zero(T) ? _opt(solver).perturb_inc_fact_first : _opt(solver).perturb_inc_fact))
            if _del_w(solver)>_opt(solver).max_hessian_perturbation
                _cnt(solver).k+=1
                @debug(_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        _del_c!(solver, (num_zero == 0 ? zero(T) : _opt(solver).jacobian_regularization_value * _mu(solver)^(_opt(solver).jacobian_regularization_exponent)))
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

    _del_w(solver) != 0 && (_del_w_last!(solver, _del_w(solver)))
    return true
end

function inertia_correction!(
    inertia_corrector::InertiaFree,
    solver::AbstractMadNLPSolver{T}
    ) where T

    n_trial = 0
    del_w_prev = zero(T)
    del_c_prev = zero(T)
    _del_w!(solver, zero(T))
    _del_c!(solver, zero(T))

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
            _del_w!(solver, _del_w_last(solver)==.0 ? _opt(solver).first_hessian_perturbation :
                max(_opt(solver).min_hessian_perturbation,_opt(solver).perturb_dec_fact*_del_w_last(solver))
                    )
        else
            _del_w!(solver, _del_w(solver) * (_del_w_last(solver)==.0 ? _opt(solver).perturb_inc_fact_first : _opt(solver).perturb_inc_fact))
            if _del_w(solver)>_opt(solver).max_hessian_perturbation
                _cnt(solver).k+=1
                @debug(_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        _del_c!(solver, _opt(solver).jacobian_regularization_value * _mu(solver)^(_opt(solver).jacobian_regularization_exponent))
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

    _del_w(solver) != 0 && (_del_w_last!(solver, _del_w(solver)))
    return true
end

function inertia_correction!(
    inertia_corrector::InertiaIgnore,
    solver::AbstractMadNLPSolver{T}
    ) where T

    n_trial = 0
    del_w_prev = zero(T)
    del_c_prev = zero(T)
    _del_w!(solver, zero(T))
    _del_c!(solver, zero(T))

    @trace(_logger(solver),"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    solve_status = solve_refine_wrapper!(
        _d(solver), solver, _p(solver), __w4(solver),
    )
    while !solve_status
        @debug(_logger(solver),"Primal-dual perturbed.")
        if n_trial == 0
            _del_w!(solver, _del_w_last(solver)==zero(T) ? _opt(solver).first_hessian_perturbation :
                max(_opt(solver).min_hessian_perturbation,_opt(solver).perturb_dec_fact*_del_w_last(solver)))
        else
            _del_w!(solver, _del_w(solver) * (_del_w_last(solver)==zero(T) ? _opt(solver).perturb_inc_fact_first : _opt(solver).perturb_inc_fact))
            if _del_w(solver)>_opt(solver).max_hessian_perturbation
                _cnt(solver).k+=1
                @debug(_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        _del_c!(solver, _opt(solver).jacobian_regularization_value * _mu(solver)^(_opt(solver).jacobian_regularization_exponent))
        regularize_diagonal!(_kkt(solver), _del_w(solver) - del_w_prev, _del_c(solver) - del_c_prev)
        del_w_prev = _del_w(solver)
        del_c_prev = _del_c(solver)

        factorize_wrapper!(solver)
        solve_status = solve_refine_wrapper!(
            _d(solver), solver, _p(solver), __w4(solver)
        )
        n_trial += 1
    end
    _del_w(solver) != 0 && (_del_w_last!(solver, _del_w(solver)))
    return true
end

function curv_test(t,n,g,kkt,wx,inertia_free_tol)
    mul_hess_blk!(wx, kkt, t)
    dot(wx,t) + max(dot(wx,n)-dot(g,n),0) - inertia_free_tol*dot(t,t) >=0
end
