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
    solver.nlp, solver;
    kwargs...)


function initialize!(solver::AbstractMadNLPSolver{T}) where T

    nlp = solver.nlp
    opt = solver.opt

    # Initializing variables
    @trace(solver.logger,"Initializing variables.")
    initialize!(
        solver.cb,
        solver.x,
        solver.xl,
        solver.xu,
        solver.y,
        solver.rhs,
        solver.ind_ineq;
        tol=opt.tol,
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


    @trace(solver.logger,"Initializing constraint duals.")
    if !solver.opt.dual_initialized
        initialize_dual(solver, opt.dual_initialization_method)
    end

    # Initializing
    solver.obj_val = eval_f_wrapper(solver, solver.x)
    eval_cons_wrapper!(solver, solver.c, solver.x)
    eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)

    theta = get_theta(solver.c)
    solver.theta_max = 1e4*max(1,theta)
    solver.theta_min = 1e-4*max(1,theta)
    solver.mu = solver.opt.mu_init
    solver.tau = max(solver.opt.tau_min,1-solver.opt.mu_init)
    push!(solver.filter, (solver.theta_max,-Inf))

    return REGULAR
end

abstract type DualInitializeOptions end
struct DualInitializeSetZero <: DualInitializeOptions end
struct DualInitializeLeastSquares <: DualInitializeOptions end

function initialize_dual(solver::MadNLPSolver{T}, ::Type{DualInitializeSetZero}) where T
    fill!(solver.y, zero(T))
end
function initialize_dual(solver::MadNLPSolver{T}, ::Type{DualInitializeLeastSquares}) where T
    set_initial_rhs!(solver, solver.kkt)
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

    solver.obj_val = eval_f_wrapper(solver, solver.x)
    eval_grad_f_wrapper!(solver, solver.f, solver.x)
    eval_cons_wrapper!(solver, solver.c, solver.x)
    eval_jac_wrapper!(solver, solver.kkt, solver.x)
    eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)

    theta = get_theta(solver.c)
    solver.theta_max=1e4*max(1,theta)
    solver.theta_min=1e-4*max(1,theta)
    solver.mu=solver.opt.mu_init
    solver.tau=max(solver.opt.tau_min,1-solver.opt.mu_init)
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
        full(solver.x)[1:get_nvar(nlp)] .= x
    end
    if y != nothing
        solver.y[1:get_ncon(nlp)] .= y
    end
    if zl != nothing
        full(solver.zl)[1:get_nvar(nlp)] .= zl
    end
    if zu != nothing
        full(solver.zu)[1:get_nvar(nlp)] .= zu
    end

    if !isempty(kwargs)
        @warn(solver.logger,"The options set during resolve may not have an effect")
        set_options!(solver.opt, kwargs)
    end

    try
        if solver.status == INITIAL
            @notice(solver.logger,"This is $(introduce()), running with $(introduce(solver.kkt.linear_solver))\n")
            print_init(solver)
            solver.status = initialize!(solver)
        else # resolving the problem
            solver.status = reinitialize!(solver)
        end

        while solver.status >= REGULAR
            solver.status == REGULAR && (solver.status = regular!(solver))
            solver.status == RESTORE && (solver.status = restore!(solver))
            solver.status == ROBUST && (solver.status = robust!(solver))
        end
    catch e
        if e isa InvalidNumberException
            if e.callback == :obj
                solver.status=INVALID_NUMBER_OBJECTIVE
            elseif e.callback == :grad
                solver.status=INVALID_NUMBER_GRADIENT
            elseif e.callback == :cons
                solver.status=INVALID_NUMBER_CONSTRAINTS
            elseif e.callback == :jac
                solver.status=INVALID_NUMBER_JACOBIAN
            elseif e.callback == :hess
                solver.status=INVALID_NUMBER_HESSIAN_LAGRANGIAN
            else
                solver.status=INVALID_NUMBER_DETECTED
            end
        elseif e isa NotEnoughDegreesOfFreedomException
            solver.status=NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa LinearSolverException
            solver.status=ERROR_IN_STEP_COMPUTATION;
            solver.opt.rethrow_error && rethrow(e)
        elseif e isa InterruptException
            solver.status=USER_REQUESTED_STOP
            solver.opt.rethrow_error && rethrow(e)
        else
            solver.status=INTERNAL_ERROR
            solver.opt.rethrow_error && rethrow(e)
        end
    finally
        solver.cnt.total_time = time() - solver.cnt.start_time
        if !(solver.status < SOLVE_SUCCEEDED)
            print_summary(solver)
        end
        @notice(solver.logger,"EXIT: $(get_status_output(solver.status, solver.opt))")
        solver.opt.disable_garbage_collector &&
            (GC.enable(true); @warn(solver.logger,"Julia garbage collector is turned back on"))
        finalize(solver.logger)

        update!(stats,solver)
    end


    return stats
end


function regular!(solver::AbstractMadNLPSolver{T}) where T
    while true
        if (solver.cnt.k!=0 && !solver.opt.jacobian_constant)
            eval_jac_wrapper!(solver, solver.kkt, solver.x)
        end
        jtprod!(solver.jacl, solver.kkt, solver.y)
        sd = get_sd(solver.y,solver.zl_r,solver.zu_r,T(solver.opt.s_max))
        sc = get_sc(solver.zl_r,solver.zu_r,T(solver.opt.s_max))
        solver.inf_pr = get_inf_pr(solver.c)
        solver.inf_du = get_inf_du(
            full(solver.f),
            full(solver.zl),
            full(solver.zu),
            solver.jacl,
            sd,
        )
        solver.inf_compl = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,zero(T),sc)
        inf_compl_mu = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,solver.mu,sc)

        print_iter(solver)

        # evaluate termination criteria
        @trace(solver.logger,"Evaluating termination criteria.")
        max(solver.inf_pr,solver.inf_du,solver.inf_compl) <= solver.opt.tol && return SOLVE_SUCCEEDED
        max(solver.inf_pr,solver.inf_du,solver.inf_compl) <= solver.opt.acceptable_tol ?
            (solver.cnt.acceptable_cnt < solver.opt.acceptable_iter ?
            solver.cnt.acceptable_cnt+=1 : return SOLVED_TO_ACCEPTABLE_LEVEL) : (solver.cnt.acceptable_cnt = 0)
        max(solver.inf_pr,solver.inf_du,solver.inf_compl) >= solver.opt.diverging_iterates_tol && return DIVERGING_ITERATES
        solver.cnt.k>=solver.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        @trace(solver.logger,"Updating the barrier parameter.")
        while solver.mu != max(solver.opt.mu_min,solver.opt.tol/10) &&
            max(solver.inf_pr,solver.inf_du,inf_compl_mu) <= solver.opt.barrier_tol_factor*solver.mu
            mu_new = get_mu(solver.mu,solver.opt.mu_min,
                            solver.opt.mu_linear_decrease_factor,solver.opt.mu_superlinear_decrease_power,solver.opt.tol)
            inf_compl_mu = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,solver.mu,sc)
            solver.tau= get_tau(solver.mu,solver.opt.tau_min)
            solver.mu = mu_new
            empty!(solver.filter)
            push!(solver.filter,(solver.theta_max,-Inf))
        end

        # compute the newton step
        @trace(solver.logger,"Computing the newton step.")
        if (solver.cnt.k!=0 && !solver.opt.hessian_constant)
            eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)
        end

        set_aug_diagonal!(solver.kkt,solver)
        set_aug_rhs!(solver, solver.kkt, solver.c)
        dual_inf_perturbation!(primal(solver.p),solver.ind_llb,solver.ind_uub,solver.mu,solver.opt.kappa_d)

        inertia_correction!(solver.inertia_corrector, solver) || return ROBUST

        # filter start
        @trace(solver.logger,"Backtracking line search initiated.")
        theta = get_theta(solver.c)
        varphi= get_varphi(solver.obj_val,solver.x_lr,solver.xl_r,solver.xu_r,solver.x_ur,solver.mu)
        varphi_d = get_varphi_d(
            primal(solver.f),
            primal(solver.x),
            primal(solver.xl),
            primal(solver.xu),
            primal(solver.d),
            solver.mu,
        )

        alpha_max = get_alpha_max(
            primal(solver.x),
            primal(solver.xl),
            primal(solver.xu),
            primal(solver.d),
            solver.tau,
        )
        solver.alpha_z = get_alpha_z(solver.zl_r,solver.zu_r,dual_lb(solver.d),dual_ub(solver.d),solver.tau)
        alpha_min = get_alpha_min(theta,varphi_d,solver.theta_min,solver.opt.gamma_theta,solver.opt.gamma_phi,
                                  solver.opt.alpha_min_frac,solver.opt.delta,solver.opt.s_theta,solver.opt.s_phi)
        solver.cnt.l = 1
        solver.alpha = alpha_max
        varphi_trial= zero(T)
        theta_trial = zero(T)
        small_search_norm = get_rel_search_norm(primal(solver.x), primal(solver.d)) < 10*eps(T)
        switching_condition = is_switching(varphi_d,solver.alpha,solver.opt.s_phi,solver.opt.delta,2.,solver.opt.s_theta)
        armijo_condition = false
        unsuccessful_iterate = false

        while true

            copyto!(full(solver.x_trial), full(solver.x))
            axpy!(solver.alpha, primal(solver.d), primal(solver.x_trial))
            solver.obj_val_trial = eval_f_wrapper(solver, solver.x_trial)
            eval_cons_wrapper!(solver, solver.c_trial, solver.x_trial)

            theta_trial = get_theta(solver.c_trial)
            varphi_trial= get_varphi(solver.obj_val_trial,solver.x_trial_lr,solver.xl_r,solver.xu_r,solver.x_trial_ur,solver.mu)
            armijo_condition = is_armijo(varphi_trial,varphi,solver.opt.eta_phi,solver.alpha,varphi_d)

            small_search_norm && break

            solver.ftype = get_ftype(
                solver.filter,theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
                solver.theta_min,solver.opt.obj_max_inc,solver.opt.gamma_theta,solver.opt.gamma_phi,
                has_constraints(solver))

            if solver.ftype in ["f","h"]
                @trace(solver.logger,"Step accepted with type $(solver.ftype)")
                break
            end

            if solver.cnt.l==1 && theta_trial>=theta
                if second_order_correction(
                    solver,alpha_max,theta,varphi,theta_trial,varphi_d,switching_condition
                    )
                    break
                end
            end

            unsuccessful_iterate = true
            solver.alpha /= 2
            solver.cnt.l += 1
            if solver.alpha < alpha_min
                @debug(solver.logger,
                       "Cannot find an acceptable step at iteration $(solver.cnt.k). Switching to restoration phase.")
                solver.cnt.k+=1
                return RESTORE
            else
                @trace(solver.logger,"Step rejected; proceed with the next trial step.")
                if solver.alpha * norm(primal(solver.d)) < eps(T)*10
                    if (solver.cnt.restoration_fail_count += 1) >= 4
                        return solver.cnt.acceptable_cnt >0 ?
                            SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
                    else
                        # (experimental) while giving up directly
                        # we give MadNLP.jl second chance to explore
                        # some possibility at the current iterate

                        fill!(solver.y, zero(T))
                        fill!(solver.zl_r, one(T))
                        fill!(solver.zu_r, one(T))
                        empty!(solver.filter)
                        push!(solver.filter,(solver.theta_max,-Inf))
                        solver.cnt.k+=1

                        return REGULAR
                    end
                end
            end
        end

        # this implements the heuristics in Section 3.2 of Ipopt paper.
        # Case I is only implemented
        if unsuccessful_iterate
            if (solver.cnt.unsuccessful_iterate += 1) >= 4
                if solver.theta_max/10 > theta_trial
                    @debug(solver.logger, "restarting filter")
                    solver.theta_max /= 10
                    empty!(solver.filter)
                    push!(solver.filter,(solver.theta_max,-Inf))
                end
                solver.cnt.unsuccessful_iterate = 0
            end
        else
            solver.cnt.unsuccessful_iterate = 0
        end

        @trace(solver.logger,"Updating primal-dual variables.")
        copyto!(full(solver.x), full(solver.x_trial))
        copyto!(solver.c, solver.c_trial)
        solver.obj_val = solver.obj_val_trial
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

        if !switching_condition || !armijo_condition
            @trace(solver.logger,"Augmenting filter.")
            augment_filter!(solver.filter,theta_trial,varphi_trial,solver.opt.gamma_theta)
        end

        solver.cnt.k+=1
            @trace(solver.logger,"Proceeding to the next interior point iteration.")
    end
end


function restore!(solver::AbstractMadNLPSolver{T}) where T
    solver.del_w = 0
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
    solver.alpha_z = zero(T)
    solver.ftype = "R"
    while true
        alpha_max = get_alpha_max(
            primal(solver.x),
            primal(solver.xl),
            primal(solver.xu),
            primal(solver.d),
            solver.tau,
        )
        solver.alpha = min(
            alpha_max,
            get_alpha_z(solver.zl_r,solver.zu_r,dual_lb(solver.d),dual_ub(solver.d),solver.tau),
            )

        axpy!(solver.alpha, primal(solver.d), full(solver.x))
        axpy!(solver.alpha, dual(solver.d), solver.y)
        solver.zl_r .+= solver.alpha .* dual_lb(solver.d)
        solver.zu_r .+= solver.alpha .* dual_ub(solver.d)

        eval_cons_wrapper!(solver,solver.c,solver.x)
        eval_grad_f_wrapper!(solver,solver.f,solver.x)
        solver.obj_val = eval_f_wrapper(solver,solver.x)

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
        solver.inf_pr = get_inf_pr(solver.c)
        solver.inf_du = get_inf_du(
            primal(solver.f),
            primal(solver.zl),
            primal(solver.zu),
            solver.jacl,
            sd,
        )

        solver.inf_compl = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,zero(T),sc)
        inf_compl_mu = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,solver.mu,sc)
        print_iter(solver)

        !solver.opt.hessian_constant && eval_lag_hess_wrapper!(solver,solver.kkt,solver.x,solver.y)
        set_aug_diagonal!(solver.kkt,solver)
        set_aug_rhs!(solver, solver.kkt, solver.c)

        dual_inf_perturbation!(primal(solver.p),solver.ind_llb,solver.ind_uub,solver.mu,solver.opt.kappa_d)
        factorize_wrapper!(solver)
        solve_refine_wrapper!(
            solver.d, solver, solver.p, solver._w4
        )

        solver.ftype = "f"
    end
end

function robust!(solver::MadNLPSolver{T}) where T
    initialize_robust_restorer!(solver)
    RR = solver.RR
    while true
        if !solver.opt.jacobian_constant
            eval_jac_wrapper!(solver, solver.kkt, solver.x)
        end
        jtprod!(solver.jacl, solver.kkt, solver.y)

        # evaluate termination criteria
        @trace(solver.logger,"Evaluating restoration phase termination criteria.")
        sd = get_sd(solver.y,solver.zl_r,solver.zu_r,solver.opt.s_max)
        sc = get_sc(solver.zl_r,solver.zu_r,solver.opt.s_max)
        solver.inf_pr = get_inf_pr(solver.c)
        solver.inf_du = get_inf_du(
            primal(solver.f),
            primal(solver.zl),
            primal(solver.zu),
            solver.jacl,
            sd,
        )
        solver.inf_compl = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,zero(T),sc)

        # Robust restoration phase error
        RR.inf_pr_R = get_inf_pr_R(solver.c,RR.pp,RR.nn)
        RR.inf_du_R = get_inf_du_R(RR.f_R,solver.y,primal(solver.zl),primal(solver.zu),solver.jacl,RR.zp,RR.zn,solver.opt.rho,sd)
        RR.inf_compl_R = get_inf_compl_R(
            solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,zero(T),sc)
        inf_compl_mu_R = get_inf_compl_R(
            solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,sc)

        print_iter(solver;is_resto=true)

        max(RR.inf_pr_R,RR.inf_du_R,RR.inf_compl_R) <= solver.opt.tol && return INFEASIBLE_PROBLEM_DETECTED
        solver.cnt.k>=solver.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        @trace(solver.logger,"Updating restoration phase barrier parameter.")
        while RR.mu_R >= solver.opt.mu_min &&
            max(RR.inf_pr_R,RR.inf_du_R,inf_compl_mu_R) <= solver.opt.barrier_tol_factor*RR.mu_R
            RR.mu_R = get_mu(RR.mu_R,solver.opt.mu_min,
                             solver.opt.mu_linear_decrease_factor,solver.opt.mu_superlinear_decrease_power,solver.opt.tol)
            inf_compl_mu_R = get_inf_compl_R(
                solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,sc)
            RR.tau_R= max(solver.opt.tau_min,1-RR.mu_R)
            RR.zeta = sqrt(RR.mu_R)

            empty!(RR.filter)
            push!(RR.filter,(solver.theta_max,-Inf))
        end

        # compute the newton step
        if !solver.opt.hessian_constant
            eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y; is_resto=true)
        end
        set_aug_RR!(solver.kkt, solver, RR)

        # without inertia correction,
        @trace(solver.logger,"Solving restoration phase primal-dual system.")
        set_aug_rhs_RR!(solver, solver.kkt, RR, solver.opt.rho)

        inertia_correction!(solver.inertia_corrector, solver) || return RESTORATION_FAILED


        finish_aug_solve_RR!(
            RR.dpp,RR.dnn,RR.dzp,RR.dzn,solver.y,dual(solver.d),
            RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,solver.opt.rho
        )


        theta_R = get_theta_R(solver.c,RR.pp,RR.nn)
        varphi_R = get_varphi_R(RR.obj_val_R,solver.x_lr,solver.xl_r,solver.xu_r,solver.x_ur,RR.pp,RR.nn,RR.mu_R)
        varphi_d_R = get_varphi_d_R(
            RR.f_R,
            primal(solver.x),
            primal(solver.xl),
            primal(solver.xu),
            primal(solver.d),
            RR.pp,RR.nn,RR.dpp,RR.dnn,RR.mu_R,solver.opt.rho,
        )

        # set alpha_min
        alpha_max = get_alpha_max_R(
            primal(solver.x),
            primal(solver.xl),
            primal(solver.xu),
            primal(solver.d),
            RR.pp,RR.dpp,RR.nn,RR.dnn,RR.tau_R,
        )
        solver.alpha_z = get_alpha_z_R(solver.zl_r,solver.zu_r,dual_lb(solver.d),dual_ub(solver.d),RR.zp,RR.dzp,RR.zn,RR.dzn,RR.tau_R)
        alpha_min = get_alpha_min(theta_R,varphi_d_R,solver.theta_min,solver.opt.gamma_theta,solver.opt.gamma_phi,
                                  solver.opt.alpha_min_frac,solver.opt.delta,solver.opt.s_theta,solver.opt.s_phi)

        # filter start
        @trace(solver.logger,"Backtracking line search initiated.")
        solver.alpha = alpha_max
        solver.cnt.l = 1
        theta_R_trial = zero(T)
        varphi_R_trial = zero(T)
        small_search_norm = get_rel_search_norm(primal(solver.x), primal(solver.d)) < 10*eps(T)
        switching_condition = is_switching(varphi_d_R,solver.alpha,solver.opt.s_phi,solver.opt.delta,theta_R,solver.opt.s_theta)
        armijo_condition = false

        while true
            copyto!(full(solver.x_trial), full(solver.x))
            copyto!(RR.pp_trial,RR.pp)
            copyto!(RR.nn_trial,RR.nn)
            axpy!(solver.alpha,primal(solver.d),primal(solver.x_trial))
            axpy!(solver.alpha,RR.dpp,RR.pp_trial)
            axpy!(solver.alpha,RR.dnn,RR.nn_trial)

            RR.obj_val_R_trial = get_obj_val_R(
                RR.pp_trial,RR.nn_trial,RR.D_R,primal(solver.x_trial),RR.x_ref,solver.opt.rho,RR.zeta)
            eval_cons_wrapper!(solver, solver.c_trial, solver.x_trial)
            theta_R_trial  = get_theta_R(solver.c_trial,RR.pp_trial,RR.nn_trial)
            varphi_R_trial = get_varphi_R(
                RR.obj_val_R_trial,solver.x_trial_lr,solver.xl_r,solver.xu_r,solver.x_trial_ur,RR.pp_trial,RR.nn_trial,RR.mu_R)

            armijo_condition = is_armijo(varphi_R_trial,varphi_R,solver.opt.eta_phi,solver.alpha,varphi_d_R)

            small_search_norm && break
            solver.ftype = get_ftype(
                RR.filter,theta_R,theta_R_trial,varphi_R,varphi_R_trial,
                switching_condition,armijo_condition,
                solver.theta_min,solver.opt.obj_max_inc,solver.opt.gamma_theta,solver.opt.gamma_phi,
                has_constraints(solver))
            solver.ftype in ["f","h"] && (@trace(solver.logger,"Step accepted with type $(solver.ftype)"); break)

            solver.alpha /= 2
            solver.cnt.l += 1
            if solver.alpha < alpha_min
                @debug(solver.logger,"Restoration phase cannot find an acceptable step at iteration $(solver.cnt.k).")
                if (solver.cnt.restoration_fail_count += 1) >= 4
                    return RESTORATION_FAILED
                else
                    # (experimental) while giving up directly
                    # we give MadNLP.jl second chance to explore
                    # some possibility at the current iterate

                    fill!(solver.y, zero(T))
                    fill!(solver.zl_r, one(T))
                    fill!(solver.zu_r, one(T))
                    empty!(solver.filter)
                    push!(solver.filter,(solver.theta_max,-Inf))

                    solver.cnt.k+=1
                    solver.cnt.t+=1
                    return REGULAR
                end
            else
                @trace(solver.logger,"Step rejected; proceed with the next trial step.")
                solver.alpha < eps(T)*10 && return solver.cnt.acceptable_cnt >0 ?
                    SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
            end
        end

        @trace(solver.logger,"Updating primal-dual variables.")
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

        if !switching_condition || !armijo_condition
            @trace(solver.logger,"Augmenting restoration phase filter.")
            augment_filter!(RR.filter,theta_R_trial,varphi_R_trial,solver.opt.gamma_theta)
        end

        # check if going back to regular phase
        @trace(solver.logger,"Checking if going back to regular phase.")
        solver.obj_val = eval_f_wrapper(solver, solver.x)
        eval_grad_f_wrapper!(solver, solver.f, solver.x)
        theta = get_theta(solver.c)
        varphi= get_varphi(solver.obj_val,solver.x_lr,solver.xl_r,solver.xu_r,solver.x_ur,solver.mu)

        if is_filter_acceptable(solver.filter,theta,varphi) &&
            theta <= solver.opt.required_infeasibility_reduction * RR.theta_ref

            @trace(solver.logger,"Going back to the regular phase.")
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

        @trace(solver.logger,"Proceeding to the next restoration phase iteration.")
        solver.cnt.k+=1
        solver.cnt.t+=1
    end
end

function second_order_correction(solver::AbstractMadNLPSolver,alpha_max,theta,varphi,
                                 theta_trial,varphi_d,switching_condition::Bool)
    @trace(solver.logger,"Second-order correction started.")

    wx = primal(solver._w1)
    wy = dual(solver._w1)
    copyto!(wy, solver.c_trial)
    axpy!(alpha_max, solver.c, wy)

    theta_soc_old = theta_trial
    for p=1:solver.opt.max_soc
        # compute second order correction
        set_aug_rhs!(solver, solver.kkt, wy)
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
        solver.obj_val_trial = eval_f_wrapper(solver, solver.x_trial)

        theta_soc = get_theta(solver.c_trial)
        varphi_soc= get_varphi(solver.obj_val_trial,solver.x_trial_lr,solver.xl_r,solver.xu_r,solver.x_trial_ur,solver.mu)

        !is_filter_acceptable(solver.filter,theta_soc,varphi_soc) && break

        if theta <=solver.theta_min && switching_condition
            # Case I
            if is_armijo(varphi_soc,varphi,solver.opt.eta_phi,solver.alpha,varphi_d)
                @trace(solver.logger,"Step in second order correction accepted by armijo condition.")
                solver.ftype = "F"
                solver.alpha=alpha_soc
                return true
            end
        else
            # Case II
            if is_sufficient_progress(theta_soc,theta,solver.opt.gamma_theta,varphi_soc,varphi,solver.opt.gamma_phi,has_constraints(solver))
                @trace(solver.logger,"Step in second order correction accepted by sufficient progress.")
                solver.ftype = "H"
                solver.alpha=alpha_soc
                return true
            end
        end

        theta_soc>solver.opt.kappa_soc*theta_soc_old && break
        theta_soc_old = theta_soc
    end
    @trace(solver.logger,"Second-order correction terminated.")

    return false
end


function inertia_correction!(
    inertia_corrector::InertiaBased,
    solver::MadNLPSolver{T}
    ) where {T}

    n_trial = 0
    solver.del_w = del_w_prev = zero(T)

    @trace(solver.logger,"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    num_pos,num_zero,num_neg = inertia(solver.kkt.linear_solver)


    solve_status = !is_inertia_correct(solver.kkt, num_pos, num_zero, num_neg) ?
        false : solve_refine_wrapper!(
            solver.d, solver, solver.p, solver._w4,
        )


    while !solve_status
        @debug(solver.logger,"Primal-dual perturbed.")

        if n_trial == 0
            solver.del_w = solver.del_w_last==zero(T) ? solver.opt.first_hessian_perturbation :
                max(solver.opt.min_hessian_perturbation,solver.opt.perturb_dec_fact*solver.del_w_last)
        else
            solver.del_w*= solver.del_w_last==zero(T) ? solver.opt.perturb_inc_fact_first : solver.opt.perturb_inc_fact
            if solver.del_w>solver.opt.max_hessian_perturbation
                solver.cnt.k+=1
                @debug(solver.logger,"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        solver.del_c = num_neg == 0 ? zero(T) : solver.opt.jacobian_regularization_value * solver.mu^(solver.opt.jacobian_regularization_exponent)
        regularize_diagonal!(solver.kkt, solver.del_w - del_w_prev, solver.del_c)
        del_w_prev = solver.del_w

        factorize_wrapper!(solver)
        num_pos,num_zero,num_neg = inertia(solver.kkt.linear_solver)

        solve_status = !is_inertia_correct(solver.kkt, num_pos, num_zero, num_neg) ?
            false : solve_refine_wrapper!(
                solver.d, solver, solver.p, solver._w4
            )
        n_trial += 1
    end

    solver.del_w != 0 && (solver.del_w_last = solver.del_w)
    return true
end

function inertia_correction!(
    inertia_corrector::InertiaFree,
    solver::MadNLPSolver{T}
    ) where T

    n_trial = 0
    solver.del_w = del_w_prev = zero(T)

    @trace(solver.logger,"Inertia-free regularization started.")
    dx = primal(solver.d)
    p0 = inertia_corrector.p0
    d0 = inertia_corrector.d0
    t = inertia_corrector.t
    n = primal(d0)
    wx= inertia_corrector.wx
    g = inertia_corrector.g

    set_g_ifr!(solver,g)
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
        @debug(solver.logger,"Primal-dual perturbed.")
        if n_trial == 0
            solver.del_w = solver.del_w_last==.0 ? solver.opt.first_hessian_perturbation :
                max(solver.opt.min_hessian_perturbation,solver.opt.perturb_dec_fact*solver.del_w_last)
        else
            solver.del_w*= solver.del_w_last==.0 ? solver.opt.perturb_inc_fact_first : solver.opt.perturb_inc_fact
            if solver.del_w>solver.opt.max_hessian_perturbation
                solver.cnt.k+=1
                @debug(solver.logger,"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        solver.del_c = solver.opt.jacobian_regularization_value * solver.mu^(solver.opt.jacobian_regularization_exponent)
        regularize_diagonal!(solver.kkt, solver.del_w - del_w_prev, solver.del_c)
        del_w_prev = solver.del_w

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

    solver.del_w != 0 && (solver.del_w_last = solver.del_w)
    return true
end

function inertia_correction!(
    inertia_corrector::InertiaIgnore,
    solver::MadNLPSolver{T}
    ) where T

    n_trial = 0
    solver.del_w = del_w_prev = zero(T)

    @trace(solver.logger,"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    solve_status = solve_refine_wrapper!(
        solver.d, solver, solver.p, solver._w4,
    )
    while !solve_status
        @debug(solver.logger,"Primal-dual perturbed.")
        if n_trial == 0
            solver.del_w = solver.del_w_last==zero(T) ? solver.opt.first_hessian_perturbation :
                max(solver.opt.min_hessian_perturbation,solver.opt.perturb_dec_fact*solver.del_w_last)
        else
            solver.del_w*= solver.del_w_last==zero(T) ? solver.opt.perturb_inc_fact_first : solver.opt.perturb_inc_fact
            if solver.del_w>solver.opt.max_hessian_perturbation
                solver.cnt.k+=1
                @debug(solver.logger,"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        solver.del_c = solver.opt.jacobian_regularization_value * solver.mu^(solver.opt.jacobian_regularization_exponent)
        regularize_diagonal!(solver.kkt, solver.del_w - del_w_prev, solver.del_c)
        del_w_prev = solver.del_w

        factorize_wrapper!(solver)
        solve_status = solve_refine_wrapper!(
            solver.d, solver, solver.p, solver._w4
        )
        n_trial += 1
    end
    solver.del_w != 0 && (solver.del_w_last = solver.del_w)
    return true
end

function curv_test(t,n,g,kkt,wx,inertia_free_tol)
    mul_hess_blk!(wx, kkt, t)
    dot(wx,t) + max(dot(wx,n)-dot(g,n),0) - inertia_free_tol*dot(t,t) >=0
end
