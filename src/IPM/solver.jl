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
        solver.cb,
        solver.x,
        solver.xl,
        solver.xu,
        solver.y,
        solver.rhs,
        solver.ind_ineq;
        tol=opt.bound_relax_factor,
        bound_push=opt.bound_push,
        bound_fac=opt.bound_fac,
    )
    fill!(solver.jacl, zero(T))
    fill!(solver.zl_r, one(T))
    fill!(solver.zu_r, one(T))

    # Initializing scaling factors
    if opt.nlp_scaling
        set_scaling!(
            solver.cb,
            solver.x,
            solver.xl,
            solver.xu,
            solver.y,
            solver.rhs,
            solver.ind_ineq,
            opt.nlp_scaling_max_gradient
        )
    end

    # Initializing KKT system
    initialize!(solver.kkt)

    # Initializing jacobian and gradient
    eval_jac_wrapper!(solver, solver.kkt, solver.x)
    eval_grad_f_wrapper!(solver, solver.f,solver.x)


    @trace(_logger(solver),"Initializing constraint duals.")
    if !solver.opt.dual_initialized
        initialize_dual(solver, opt.dual_initialization_method)
    end

    # Initializing
    _obj_val!(solver, eval_f_wrapper(solver, solver.x))
    eval_cons_wrapper!(solver, solver.c, solver.x)
    eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)

    theta = get_theta(solver.c)
    _theta_max!(solver, 1e4*max(1,theta))
    _theta_min!(solver, 1e-4*max(1,theta))
    _mu!(solver, solver.opt.barrier.mu_init)
    _tau!(solver, max(solver.opt.tau_min,1-solver.opt.barrier.mu_init))
    push!(solver.filter, (solver.theta_max,-Inf))

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
        solver.d, solver, solver.p, solver._w4
    )
    if !is_solved || (norm(dual(solver.d), Inf) > solver.opt.constr_mult_init_max)
        fill!(solver.y, zero(T))
    else
        copyto!(solver.y, dual(solver.d))
    end
end

function reinitialize!(solver::AbstractMadNLPSolver)
    variable(solver.x) .= get_x0(solver.nlp)

    _obj_val!(solver, eval_f_wrapper(solver, solver.x))
    eval_grad_f_wrapper!(solver, solver.f, solver.x)
    eval_cons_wrapper!(solver, solver.c, solver.x)
    eval_jac_wrapper!(solver, solver.kkt, solver.x)
    eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)

    theta = get_theta(solver.c)
    _theta_max!(solver, 1e4*max(1,theta))
    _theta_min!(solver, 1e-4*max(1,theta))
    _mu!(solver, solver.opt.barrier.mu_init)
    _tau!(solver, max(solver.opt.tau_min,1-solver.opt.barrier.mu_init))
    empty!(solver.filter)
    push!(solver.filter, (solver.theta_max,-Inf))

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
            @notice(_logger(solver),"This is $(introduce()), running with $(introduce(solver.kkt.linear_solver))\n")
            print_init(solver)
            _status!(solver, initialize!(solver))
        else # resolving the problem
            _status!(solver, reinitialize!(solver))
        end

        while solver.status >= REGULAR
            solver.status == REGULAR && (_status!(solver, regular!(solver)))
            solver.status == RESTORE && (_status!(solver, restore!(solver)))
            solver.status == ROBUST && (_status!(solver, robust!(solver)))
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
            solver.opt.rethrow_error && rethrow(e)
        elseif e isa InterruptException
            _status!(solver, USER_REQUESTED_STOP)
            solver.opt.rethrow_error && rethrow(e)
        else
            _status!(solver, INTERNAL_ERROR)
            solver.opt.rethrow_error && rethrow(e)
        end
    finally
        solver.cnt.total_time = time() - solver.cnt.start_time
        if !(solver.status < SOLVE_SUCCEEDED)
            print_summary(solver)
        end
        @notice(_logger(solver),"$(Base.text_colors[color_status(solver.status)])EXIT: $(get_status_output(solver.status, solver.opt))$(Base.text_colors[:normal])")
        solver.opt.disable_garbage_collector &&
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
        if (solver.cnt.k!=0 && !solver.opt.jacobian_constant)
            eval_jac_wrapper!(solver, solver.kkt, solver.x)
        end

        jtprod!(solver.jacl, solver.kkt, solver.y)
        sd = get_sd(solver.y,solver.zl_r,solver.zu_r,T(solver.opt.s_max))
        sc = get_sc(solver.zl_r,solver.zu_r,T(solver.opt.s_max))
        _inf_pr!(solver, get_inf_pr(solver.c))
        _inf_du!(solver, get_inf_du(
            full(solver.f),
            full(solver.zl),
            full(solver.zu),
            solver.jacl,
            sd,
        ))
        _inf_compl!(solver, get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,zero(T),sc))

        print_iter(solver)

        # evaluate termination criteria
        @trace(_logger(solver),"Evaluating termination criteria.")
        max(solver.inf_pr,solver.inf_du,solver.inf_compl) <= solver.opt.tol && return SOLVE_SUCCEEDED
        max(solver.inf_pr,solver.inf_du,solver.inf_compl) <= solver.opt.acceptable_tol ?
            (solver.cnt.acceptable_cnt < solver.opt.acceptable_iter ?
            solver.cnt.acceptable_cnt+=1 : return SOLVED_TO_ACCEPTABLE_LEVEL) : (solver.cnt.acceptable_cnt = 0)
        max(solver.inf_pr,solver.inf_du,solver.inf_compl) >= solver.opt.diverging_iterates_tol && return DIVERGING_ITERATES
        solver.cnt.k>=solver.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        # evaluate Hessian
        if (solver.cnt.k!=0 && !solver.opt.hessian_constant)
            eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)
        end

        # update the barrier parameter
        @trace(solver.logger,"Updating the barrier parameter.")
        update_barrier!(solver.opt.barrier, solver, sc)

        # factorize the KKT system and solve Newton step
        @trace(solver.logger,"Computing the Newton step.")
        set_aug_diagonal!(solver.kkt,solver)
        set_aug_rhs!(solver, solver.kkt, solver.c, solver.mu)
        dual_inf_perturbation!(primal(solver.p),solver.ind_llb,solver.ind_uub,solver.mu,solver.opt.kappa_d)
        inertia_correction!(solver.inertia_corrector, solver) || return ROBUST

        @trace(_logger(solver),"Backtracking line search initiated.")
        status = filter_line_search!(solver)
        if status != LINESEARCH_SUCCEEDED
            return status
        end

        @trace(_logger(solver),"Updating primal-dual variables.")
        copyto!(full(solver.x), full(solver.x_trial))
        copyto!(solver.c, solver.c_trial)
        _obj_val!(solver, solver.obj_val_trial)
        adjust_boundary!(solver.x_lr,solver.xl_r,solver.x_ur,solver.xu_r,solver.mu)

        axpy!(solver.alpha,dual(solver.d),solver.y)

        solver.zl_r .+= solver.alpha_z .* dual_lb(solver.d)
        solver.zu_r .+= solver.alpha_z .* dual_ub(solver.d)
        reset_bound_dual!(
            primal(solver.zl),
            primal(solver.x),
            primal(solver.xl),
            solver.mu,solver.opt.kappa_sigma,
        )
        reset_bound_dual!(
            primal(solver.zu),
            primal(solver.xu),
            primal(solver.x),
            solver.mu,solver.opt.kappa_sigma,
        )

        eval_grad_f_wrapper!(solver, solver.f,solver.x)

        solver.cnt.k+=1
        @trace(_logger(solver),"Proceeding to the next interior point iteration.")
    end
end


function restore!(solver::AbstractMadNLPSolver{T}) where T
    _del_w!(solver, 0)
    # Backup the previous primal iterate
    copyto!(primal(solver._w1), full(solver.x))
    copyto!(dual(solver._w1), solver.y)
    copyto!(dual(solver._w2), solver.c)

    F = get_F(
        solver.c,
        primal(solver.f),
        primal(solver.zl),
        primal(solver.zu),
        solver.jacl,
        solver.x_lr,
        solver.xl_r,
        solver.zl_r,
        solver.xu_r,
        solver.x_ur,
        solver.zu_r,
        solver.mu,
    )
    _alpha_z!(solver, zero(T))
    _ftype!(solver, "R")
    while true
        alpha_max = get_alpha_max(
            primal(solver.x),
            primal(solver.xl),
            primal(solver.xu),
            primal(solver.d),
            solver.tau,
        )
        _alpha!(solver,min(
            alpha_max,
            get_alpha_z(solver.zl_r,solver.zu_r,dual_lb(solver.d),dual_ub(solver.d),solver.tau),
        ))

        axpy!(solver.alpha, primal(solver.d), full(solver.x))
        axpy!(solver.alpha, dual(solver.d), solver.y)
        solver.zl_r .+= solver.alpha .* dual_lb(solver.d)
        solver.zu_r .+= solver.alpha .* dual_ub(solver.d)

        eval_cons_wrapper!(solver,solver.c,solver.x)
        eval_grad_f_wrapper!(solver,solver.f,solver.x)
        _obj_val!(solver, eval_f_wrapper(solver,solver.x))

        !solver.opt.jacobian_constant && eval_jac_wrapper!(solver,solver.kkt,solver.x)
        jtprod!(solver.jacl,solver.kkt,solver.y)

        F_trial = get_F(
            solver.c,
            primal(solver.f),
            primal(solver.zl),
            primal(solver.zu),
            solver.jacl,
            solver.x_lr,
            solver.xl_r,
            solver.zl_r,
            solver.xu_r,
            solver.x_ur,
            solver.zu_r,
            solver.mu,
        )
        if F_trial > solver.opt.soft_resto_pderror_reduction_factor*F
            copyto!(primal(solver.x), primal(solver._w1))
            copyto!(solver.y, dual(solver._w1))
            copyto!(solver.c, dual(solver._w2)) # backup the previous primal iterate
            return ROBUST
        end

        adjust_boundary!(solver.x_lr,solver.xl_r,solver.x_ur,solver.xu_r,solver.mu)

        F = F_trial

        theta = get_theta(solver.c)
        varphi= get_varphi(solver.obj_val,solver.x_lr,solver.xl_r,solver.xu_r,solver.x_ur,solver.mu)

        solver.cnt.k+=1

        is_filter_acceptable(solver.filter,theta,varphi) ? (return REGULAR) : (solver.cnt.t+=1)
        solver.cnt.k>=solver.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED


        sd = get_sd(solver.y,solver.zl_r,solver.zu_r,solver.opt.s_max)
        sc = get_sc(solver.zl_r,solver.zu_r,solver.opt.s_max)
        _inf_pr!(solver, get_inf_pr(solver.c))
        _inf_du!(solver, get_inf_du(
            primal(solver.f),
            primal(solver.zl),
            primal(solver.zu),
            solver.jacl,
            sd,
        ))

        _inf_compl!(solver, get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,zero(T),sc))
        inf_compl_mu = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,solver.mu,sc)
        print_iter(solver)

        !solver.opt.hessian_constant && eval_lag_hess_wrapper!(solver,solver.kkt,solver.x,solver.y)
        set_aug_diagonal!(solver.kkt,solver)
        set_aug_rhs!(solver, solver.kkt, solver.c, solver.mu)

        dual_inf_perturbation!(primal(solver.p),solver.ind_llb,solver.ind_uub,solver.mu,solver.opt.kappa_d)
        factorize_wrapper!(solver)
        solve_refine_wrapper!(
            solver.d, solver, solver.p, solver._w4
        )

        _ftype!(solver, "f")
    end
end

function robust!(solver::AbstractMadNLPSolver{T}) where T
    initialize_robust_restorer!(solver)
    RR = solver.RR
    while true
        if !solver.opt.jacobian_constant
            eval_jac_wrapper!(solver, solver.kkt, solver.x)
        end
        jtprod!(solver.jacl, solver.kkt, solver.y)

        # evaluate termination criteria
        @trace(_logger(solver),"Evaluating restoration phase termination criteria.")
        sd = get_sd(solver.y,solver.zl_r,solver.zu_r,solver.opt.s_max)
        sc = get_sc(solver.zl_r,solver.zu_r,solver.opt.s_max)
        _inf_pr!(solver, get_inf_pr(solver.c))
        _inf_du!(get_inf_du(
            primal(solver.f),
            primal(solver.zl),
            primal(solver.zu),
            solver.jacl,
            sd,
        ))
        _inf_compl!(solver, get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,zero(T),sc))

        # Robust restoration phase error
        RR.inf_pr_R = get_inf_pr_R(solver.c,RR.pp,RR.nn)
        RR.inf_du_R = get_inf_du_R(RR.f_R,solver.y,primal(solver.zl),primal(solver.zu),solver.jacl,RR.zp,RR.zn,solver.opt.rho,sd)
        RR.inf_compl_R = get_inf_compl_R(
            solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,zero(T),sc)

        print_iter(solver;is_resto=true)

        max(RR.inf_pr_R,RR.inf_du_R,RR.inf_compl_R) <= solver.opt.tol && return INFEASIBLE_PROBLEM_DETECTED
        solver.cnt.k>=solver.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        @trace(solver.logger,"Updating restoration phase barrier parameter.")
        _update_monotone_RR!(solver.opt.barrier, solver, sc)

        # compute the Newton step
        if !solver.opt.hessian_constant
            eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y; is_resto=true)
        end

        # without inertia correction,
        @trace(_logger(solver),"Solving restoration phase primal-dual system.")
        set_aug_rhs_RR!(solver, solver.kkt, RR, solver.opt.rho)
        inertia_correction!(solver.inertia_corrector, solver) || return RESTORATION_FAILED
        finish_aug_solve_RR!(
            RR.dpp,RR.dnn,RR.dzp,RR.dzn,solver.y,dual(solver.d),
            RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,solver.opt.rho
        )

        # filter start
        @trace(_logger(solver),"Backtracking line search initiated.")
        status = filter_line_search_RR!(solver)
        if status != LINESEARCH_SUCCEEDED
            return status
        end

        @trace(_logger(solver),"Updating primal-dual variables.")
        copyto!(full(solver.x), full(solver.x_trial))
        copyto!(solver.c, solver.c_trial)
        copyto!(RR.pp, RR.pp_trial)
        copyto!(RR.nn, RR.nn_trial)

        RR.obj_val_R=RR.obj_val_R_trial
        set_f_RR!(solver,RR)

        axpy!(solver.alpha, dual(solver.d), solver.y)
        axpy!(solver.alpha_z, RR.dzp,RR.zp)
        axpy!(solver.alpha_z, RR.dzn,RR.zn)

        solver.zl_r .+= solver.alpha_z .* dual_lb(solver.d)
        solver.zu_r .+= solver.alpha_z .* dual_ub(solver.d)

        reset_bound_dual!(
            primal(solver.zl),
            primal(solver.x),
            primal(solver.xl),
            RR.mu_R, solver.opt.kappa_sigma,
        )
        reset_bound_dual!(
            primal(solver.zu),
            primal(solver.xu),
            primal(solver.x),
            RR.mu_R, solver.opt.kappa_sigma,
        )
        reset_bound_dual!(RR.zp,RR.pp,RR.mu_R,solver.opt.kappa_sigma)
        reset_bound_dual!(RR.zn,RR.nn,RR.mu_R,solver.opt.kappa_sigma)

        adjust_boundary!(solver.x_lr,solver.xl_r,solver.x_ur,solver.xu_r,solver.mu)

        # check if going back to regular phase
        @trace(_logger(solver),"Checking if going back to regular phase.")
        _obj_val!(solver, eval_f_wrapper(solver, solver.x))
        eval_grad_f_wrapper!(solver, solver.f, solver.x)
        theta = get_theta(solver.c)
        varphi= get_varphi(solver.obj_val,solver.x_lr,solver.xl_r,solver.xu_r,solver.x_ur,solver.mu)

        if is_filter_acceptable(solver.filter,theta,varphi) &&
            theta <= solver.opt.required_infeasibility_reduction * RR.theta_ref

            @trace(_logger(solver),"Going back to the regular phase.")
            set_initial_rhs!(solver, solver.kkt)
            initialize!(solver.kkt)

            factorize_wrapper!(solver)
            solve_refine_wrapper!(
                solver.d, solver, solver.p, solver._w4
            )
            if norm(dual(solver.d), Inf)>solver.opt.constr_mult_init_max
                fill!(solver.y, zero(T))
            else
                copyto!(solver.y, dual(solver.d))
            end

            solver.cnt.k+=1
            solver.cnt.t+=1

            return REGULAR
        end

        solver.cnt.k>=solver.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        @trace(_logger(solver),"Proceeding to the next restoration phase iteration.")
        solver.cnt.k+=1
        solver.cnt.t+=1
    end
end

function second_order_correction(solver::AbstractMadNLPSolver,alpha_max,theta,varphi,
                                 theta_trial,varphi_d,switching_condition::Bool)
    @trace(_logger(solver),"Second-order correction started.")

    wx = primal(solver._w1)
    wy = dual(solver._w1)
    copyto!(wy, solver.c_trial)
    axpy!(alpha_max, solver.c, wy)

    theta_soc_old = theta_trial
    for p=1:solver.opt.max_soc
        # compute second order correction
        set_aug_rhs!(solver, solver.kkt, wy, solver.mu)
        dual_inf_perturbation!(
            primal(solver.p),
            solver.ind_llb,solver.ind_uub,solver.mu,solver.opt.kappa_d,
        )
        solve_refine_wrapper!(
            solver._w1, solver, solver.p, solver._w4
        )
        alpha_soc = get_alpha_max(
            primal(solver.x),
            primal(solver.xl),
            primal(solver.xu),
            wx,solver.tau
        )

        copyto!(primal(solver.x_trial), primal(solver.x))
        axpy!(alpha_soc, wx, primal(solver.x_trial))
        eval_cons_wrapper!(solver, solver.c_trial, solver.x_trial)
        _obj_val_trial!(solver, eval_f_wrapper(solver, solver.x_trial))

        theta_soc = get_theta(solver.c_trial)
        varphi_soc= get_varphi(solver.obj_val_trial,solver.x_trial_lr,solver.xl_r,solver.xu_r,solver.x_trial_ur,solver.mu)

        !is_filter_acceptable(solver.filter,theta_soc,varphi_soc) && break

        if theta <=solver.theta_min && switching_condition
            # Case I
            if is_armijo(varphi_soc,varphi,solver.opt.eta_phi,solver.alpha,varphi_d)
                @trace(_logger(solver),"Step in second order correction accepted by armijo condition.")
                _ftype!(solver, "F")
                _alpha!(solver, alpha_soc)
                return true
            end
        else
            # Case II
            if is_sufficient_progress(theta_soc,theta,solver.opt.gamma_theta,varphi_soc,varphi,solver.opt.gamma_phi,has_constraints(solver))
                @trace(_logger(solver),"Step in second order correction accepted by sufficient progress.")
                _ftype!(solver, "H")
                _alpha!(solver, alpha_soc)
                return true
            end
        end

        theta_soc>solver.opt.kappa_soc*theta_soc_old && break
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
    _del_w!(solver, del_w_prev = zero(T))
    _del_c!(solver, del_c_prev = zero(T))

    @trace(_logger(solver),"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    num_pos,num_zero,num_neg = inertia(solver.kkt.linear_solver)

    solve_status = if is_inertia_correct(solver.kkt, num_pos, num_zero, num_neg)
        # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
        solve_refine_wrapper!(solver.d, solver, solver.p, solver._w4)
    else
        false
    end

    while !solve_status
        @debug(_logger(solver),"Primal-dual perturbed.")

        if n_trial == 0
            _del_w!(solver, solver.del_w_last==zero(T) ? solver.opt.first_hessian_perturbation :
                max(solver.opt.min_hessian_perturbation,solver.opt.perturb_dec_fact*solver.del_w_last)
                    )
        else
            _del_w!(solver, solver.del_w * solver.del_w_last==zero(T) ? solver.opt.perturb_inc_fact_first : solver.opt.perturb_inc_fact)
            if solver.del_w>solver.opt.max_hessian_perturbation
                solver.cnt.k+=1
                @debug(_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        _del_c!(solver, num_zero == 0 ? zero(T) : solver.opt.jacobian_regularization_value * solver.mu^(solver.opt.jacobian_regularization_exponent))
        regularize_diagonal!(solver.kkt, solver.del_w - del_w_prev, solver.del_c - del_c_prev)
        del_w_prev = solver.del_w
        del_c_prev = solver.del_c

        factorize_wrapper!(solver)
        num_pos,num_zero,num_neg = inertia(solver.kkt.linear_solver)

        solve_status = if is_inertia_correct(solver.kkt, num_pos, num_zero, num_neg)
            solve_refine_wrapper!(solver.d, solver, solver.p, solver._w4)
        else
            false
        end

        n_trial += 1
    end

    solver.del_w != 0 && (_del_w_last!(solver, solver.del_w))
    return true
end

function inertia_correction!(
    inertia_corrector::InertiaFree,
    solver::AbstractMadNLPSolver{T}
    ) where T
    n_trial = 0
    _del_w!(solver, del_w_prev = zero(T))
    _del_c!(solver, del_c_prev = zero(T))

    @trace(_logger(solver),"Inertia-free regularization started.")
    dx = primal(solver.d)
    p0 = inertia_corrector.p0
    d0 = inertia_corrector.d0
    t = inertia_corrector.t
    n = primal(d0)
    wx= inertia_corrector.wx
    g = inertia_corrector.g

    set_g_ifr!(solver,g)
    # Initialize p0
    set_aug_rhs_ifr!(solver, solver.kkt, p0)

    factorize_wrapper!(solver)

    solve_status = solve_refine_wrapper!(
        d0, solver, p0, solver._w3,
    ) && solve_refine_wrapper!(
        solver.d, solver, solver.p, solver._w4,
    )
    copyto!(t,dx)
    axpy!(-1.,n,t)

    while !curv_test(t,n,g,solver.kkt,wx,solver.opt.inertia_free_tol)  || !solve_status
        @debug(_logger(solver),"Primal-dual perturbed.")
        if n_trial == 0
            _del_w!(solver, solver.del_w_last==.0 ? solver.opt.first_hessian_perturbation :
                max(solver.opt.min_hessian_perturbation,solver.opt.perturb_dec_fact*solver.del_w_last)
                    )
        else
            _del_w!(solver, solver.del_w * solver.del_w_last==.0 ? solver.opt.perturb_inc_fact_first : solver.opt.perturb_inc_fact)
            if solver.del_w>solver.opt.max_hessian_perturbation
                solver.cnt.k+=1
                @debug(_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        _del_c!(solver, solver.opt.jacobian_regularization_value * solver.mu^(solver.opt.jacobian_regularization_exponent))
        regularize_diagonal!(solver.kkt, solver.del_w - del_w_prev, solver.del_c - del_c_prev)
        del_w_prev = solver.del_w
        del_c_prev = solver.del_c

        factorize_wrapper!(solver)
        solve_status = solve_refine_wrapper!(
            d0, solver, p0, solver._w3
        ) && solve_refine_wrapper!(
            solver.d, solver, solver.p, solver._w4
        )
        copyto!(t,dx)
        axpy!(-1.,n,t)

        n_trial += 1
    end

    solver.del_w != 0 && (_del_w_last!(solver, solver.del_w))
    return true
end

function inertia_correction!(
    inertia_corrector::InertiaIgnore,
    solver::AbstractMadNLPSolver{T}
    ) where T

    n_trial = 0
    _del_w!(solver, del_w_prev = zero(T))
    _del_c!(solver, del_c_prev = zero(T))

    @trace(_logger(solver),"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    solve_status = solve_refine_wrapper!(
        solver.d, solver, solver.p, solver._w4,
    )
    while !solve_status
        @debug(_logger(solver),"Primal-dual perturbed.")
        if n_trial == 0
            _del_w!(solver, solver.del_w_last==zero(T) ? solver.opt.first_hessian_perturbation :
                max(solver.opt.min_hessian_perturbation,solver.opt.perturb_dec_fact*solver.del_w_last))
        else
            _del_w!(solver.del_w * solver.del_w_last==zero(T) ? solver.opt.perturb_inc_fact_first : solver.opt.perturb_inc_fact)
            if solver.del_w>solver.opt.max_hessian_perturbation
                solver.cnt.k+=1
                @debug(_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        _del_c!(solver, solver.opt.jacobian_regularization_value * solver.mu^(solver.opt.jacobian_regularization_exponent))
        regularize_diagonal!(solver.kkt, solver.del_w - del_w_prev, solver.del_c - del_c_prev)
        del_w_prev = solver.del_w
        del_c_prev = solver.del_c

        factorize_wrapper!(solver)
        solve_status = solve_refine_wrapper!(
            solver.d, solver, solver.p, solver._w4
        )
        n_trial += 1
    end
    solver.del_w != 0 && (_del_w_last!(solver, solver.del_w))
    return true
end

function curv_test(t,n,g,kkt,wx,inertia_free_tol)
    mul_hess_blk!(wx, kkt, t)
    dot(wx,t) + max(dot(wx,n)-dot(g,n),0) - inertia_free_tol*dot(t,t) >=0
end
