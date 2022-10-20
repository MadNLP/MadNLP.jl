function madnlp(model::AbstractNLPModel; kwargs...)
    solver = MadNLPSolver(model;kwargs...)
    initialize!(solver.kkt)
    return solve!(solver)
end

function solve!(
    model::AbstractNLPModel,
    solver::MadNLPSolver,
    nlp::AbstractNLPModel;
    x = nlp.meta.x,
    y = nlp.meta.y,
    zl= nothing,
    zu= nothing,
    kwargs...)

    @assert solver.nlp == nlp
    
    solve!(
        solver;
        x = x, y = y,
        zl = zl, zu = zu,
        kwargs...
    )
end

function initialize!(solver::AbstractMadNLPSolver)
    # initializing slack variables
    @trace(solver.logger,"Initializing slack variables.")
    cons!(solver.nlp,get_x0(solver.nlp),_madnlp_unsafe_wrap(solver.c,get_ncon(solver.nlp)))
    solver.cnt.con_cnt += 1
    solver.x_slk.=solver.c_slk

    # Initialization
    @trace(solver.logger,"Initializing primal and bound duals.")
    solver.zl_r.=1.0
    solver.zu_r.=1.0
    solver.xl_r.-= max.(1,abs.(solver.xl_r)).*solver.opt.tol
    solver.xu_r.+= max.(1,abs.(solver.xu_r)).*solver.opt.tol
    initialize_variables!(solver.x,solver.xl,solver.xu,solver.opt.bound_push,solver.opt.bound_fac)

    # Automatic scaling (constraints)
    @trace(solver.logger,"Computing constraint scaling.")
    eval_jac_wrapper!(solver, solver.kkt, solver.x)
    compress_jacobian!(solver.kkt)
    if (solver.m > 0) && solver.opt.nlp_scaling
        jac = get_raw_jacobian(solver.kkt)
        scale_constraints!(solver.nlp, solver.con_scale, jac; max_gradient=solver.opt.nlp_scaling_max_gradient)
        set_jacobian_scaling!(solver.kkt, solver.con_scale)
        solver.y./=solver.con_scale
    end
    compress_jacobian!(solver.kkt)

    # Automatic scaling (objective)
    eval_grad_f_wrapper!(solver, solver.f,solver.x)
    @trace(solver.logger,"Computing objective scaling.")
    if solver.opt.nlp_scaling
        solver.obj_scale[] = scale_objective(solver.nlp, solver.f; max_gradient=solver.opt.nlp_scaling_max_gradient)
        solver.f.*=solver.obj_scale[]
    end

    # Initialize dual variables
    @trace(solver.logger,"Initializing constraint duals.")
    if !solver.opt.dual_initialized
        set_initial_rhs!(solver, solver.kkt)
        initialize!(solver.kkt)
        factorize_wrapper!(solver)
        solve_refine_wrapper!(solver,solver.d,solver.p)
        if norm(dual(solver.d), Inf) > solver.opt.constr_mult_init_max
            fill!(solver.y, 0.0)
        else
            copyto!(solver.y, dual(solver.d))
        end
    end

    # Initializing
    solver.obj_val = eval_f_wrapper(solver, solver.x)
    eval_cons_wrapper!(solver, solver.c, solver.x)
    eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)

    theta = get_theta(solver.c)
    solver.theta_max=1e4*max(1,theta)
    solver.theta_min=1e-4*max(1,theta)
    solver.mu=solver.opt.mu_init
    solver.tau=max(solver.opt.tau_min,1-solver.opt.mu_init)
    solver.filter = [(solver.theta_max,-Inf)]

    return REGULAR
end


function reinitialize!(solver::AbstractMadNLPSolver)
    view(solver.x,1:get_nvar(solver.nlp)) .= get_x0(solver.nlp)

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
    solver.filter = [(solver.theta_max,-Inf)]

    return REGULAR
end

# major loops ---------------------------------------------------------
function solve!(
    solver::AbstractMadNLPSolver;
    x = nothing, y = nothing,
    zl = nothing, zu = nothing,
    kwargs...)

    if x != nothing
        solver.x[1:get_nvar(solver.nlp)] .= x
    end
    if y != nothing
        solver.y[1:get_nvar(solver.nlp)] .= y
    end
    if zl != nothing
        solver.zl[1:get_nvar(solver.nlp)] .= zl
    end
    if zu != nothing
        solver.zu[1:get_nvar(solver.nlp)] .= zu
    end

    if !isempty(kwargs)
        @warn(solver.logger,"The options set during resolve may not have an effect")
        set_options!(solver.opt,Dict{Symbol,Any}(),kwargs)
    end
    
    try
        if solver.status == INITIAL
            @notice(solver.logger,"This is $(introduce()), running with $(introduce(solver.linear_solver))\n")
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
            solver.status=INVALID_NUMBER_DETECTED
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
        !(solver.status < SOLVE_SUCCEEDED) && (print_summary_1(solver);print_summary_2(solver))
        # Unscale once the summary has been printed out
        unscale!(solver)
        @notice(solver.logger,"EXIT: $(STATUS_OUTPUT_DICT[solver.status])")
        solver.opt.disable_garbage_collector &&
            (GC.enable(true); @warn(solver.logger,"Julia garbage collector is turned back on"))
        finalize(solver.logger)
    end
    return MadNLPExecutionStats(solver)
end


function unscale!(solver::AbstractMadNLPSolver)
    solver.obj_val/=solver.obj_scale[]
    solver.c ./= solver.con_scale
    solver.c .+= solver.rhs
    solver.c_slk .+= solver.x_slk
end

function regular!(solver::AbstractMadNLPSolver)
    while true
        if (solver.cnt.k!=0 && !solver.opt.jacobian_constant)
            eval_jac_wrapper!(solver, solver.kkt, solver.x)
        end
        jtprod!(solver.jacl, solver.kkt, solver.y)
        fixed_variable_treatment_vec!(solver.jacl,solver.ind_fixed)
        fixed_variable_treatment_z!(solver.zl,solver.zu,solver.f,solver.jacl,solver.ind_fixed)

        sd = get_sd(solver.y,solver.zl_r,solver.zu_r,solver.opt.s_max)
        sc = get_sc(solver.zl_r,solver.zu_r,solver.opt.s_max)

        solver.inf_pr = get_inf_pr(solver.c)
        solver.inf_du = get_inf_du(solver.f,solver.zl,solver.zu,solver.jacl,sd)
        solver.inf_compl = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,0.,sc)
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
        if solver.opt.inertia_correction_method == INERTIA_FREE
            set_aug_rhs_ifr!(solver, solver.kkt)
        end
        dual_inf_perturbation!(primal(solver.p),solver.ind_llb,solver.ind_uub,solver.mu,solver.opt.kappa_d)

        # start inertia conrrection
        @trace(solver.logger,"Solving primal-dual system.")
        if solver.opt.inertia_correction_method == INERTIA_FREE
            inertia_free_reg(solver) || return ROBUST
        elseif solver.opt.inertia_correction_method == INERTIA_BASED
            inertia_based_reg(solver) || return ROBUST
        end

        finish_aug_solve!(solver, solver.kkt, solver.mu)

        # filter start
        @trace(solver.logger,"Backtracking line search initiated.")
        theta = get_theta(solver.c)
        varphi= get_varphi(solver.obj_val,solver.x_lr,solver.xl_r,solver.xu_r,solver.x_ur,solver.mu)
        varphi_d = get_varphi_d(solver.f,solver.x,solver.xl,solver.xu,primal(solver.d),solver.mu)


        alpha_max = get_alpha_max(solver.x,solver.xl,solver.xu,primal(solver.d),solver.tau)
        solver.alpha_z = get_alpha_z(solver.zl_r,solver.zu_r,dual_lb(solver.d),dual_ub(solver.d),solver.tau)
        alpha_min = get_alpha_min(theta,varphi_d,solver.theta_min,solver.opt.gamma_theta,solver.opt.gamma_phi,
                                  solver.opt.alpha_min_frac,solver.opt.delta,solver.opt.s_theta,solver.opt.s_phi)
        solver.cnt.l = 1
        solver.alpha = alpha_max
        varphi_trial= 0.
            theta_trial = 0.
            small_search_norm = get_rel_search_norm(solver.x, primal(solver.d)) < 10*eps(eltype(solver.x))
        switching_condition = is_switching(varphi_d,solver.alpha,solver.opt.s_phi,solver.opt.delta,2.,solver.opt.s_theta)
        armijo_condition = false
        while true
            copyto!(solver.x_trial, solver.x)
            axpy!(solver.alpha, primal(solver.d), solver.x_trial)

            solver.obj_val_trial = eval_f_wrapper(solver, solver.x_trial)
            eval_cons_wrapper!(solver, solver.c_trial, solver.x_trial)

            theta_trial = get_theta(solver.c_trial)
            varphi_trial= get_varphi(solver.obj_val_trial,solver.x_trial_lr,solver.xl_r,solver.xu_r,solver.x_trial_ur,solver.mu)
            armijo_condition = is_armijo(varphi_trial,varphi,solver.opt.eta_phi,solver.alpha,varphi_d)

            # println(armijo_condition)
            small_search_norm && break

            solver.ftype = get_ftype(
                solver.filter,theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
                solver.theta_min,solver.opt.obj_max_inc,solver.opt.gamma_theta,solver.opt.gamma_phi,
                has_constraints(solver))
            solver.ftype in ["f","h"] && (@trace(solver.logger,"Step accepted with type $(solver.ftype)"); break)

            solver.cnt.l==1 && theta_trial>=theta && second_order_correction(
                solver,alpha_max,theta,varphi,theta_trial,varphi_d,switching_condition) && break

            solver.alpha /= 2
            solver.cnt.l += 1
            if solver.alpha < alpha_min
                @debug(solver.logger,
                       "Cannot find an acceptable step at iteration $(solver.cnt.k). Switching to restoration phase.")
                solver.cnt.k+=1
                return RESTORE
            else
                @trace(solver.logger,"Step rejected; proceed with the next trial step.")
                solver.alpha * norm(primal(solver.d)) < eps(eltype(solver.x))*10 &&
                    return solver.cnt.acceptable_cnt >0 ?
                    SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
            end
        end

        @trace(solver.logger,"Updating primal-dual variables.")
        solver.x.=solver.x_trial
        solver.c.=solver.c_trial
        solver.obj_val=solver.obj_val_trial
        adjusted = adjust_boundary!(solver.x_lr,solver.xl_r,solver.x_ur,solver.xu_r,solver.mu)
        adjusted > 0 &&
            @warn(solver.logger,"In iteration $(solver.cnt.k), $adjusted Slack too small, adjusting variable bound")

        axpy!(solver.alpha,dual(solver.d),solver.y)
        axpy!(solver.alpha_z, dual_lb(solver.d), solver.zl_r)
        axpy!(solver.alpha_z, dual_ub(solver.d), solver.zu_r)
        reset_bound_dual!(solver.zl,solver.x,solver.xl,solver.mu,solver.opt.kappa_sigma)
        reset_bound_dual!(solver.zu,solver.xu,solver.x,solver.mu,solver.opt.kappa_sigma)
        eval_grad_f_wrapper!(solver, solver.f,solver.x)

        if !switching_condition || !armijo_condition
            @trace(solver.logger,"Augmenting filter.")
            augment_filter!(solver.filter,theta_trial,varphi_trial,solver.opt.gamma_theta)
        end

        solver.cnt.k+=1
        @trace(solver.logger,"Proceeding to the next interior point iteration.")
    end
end

function restore!(solver::AbstractMadNLPSolver)
    solver.del_w=0
    primal(solver._w1) .= solver.x # backup the previous primal iterate
    dual(solver._w1) .= solver.y # backup the previous primal iterate
    dual(solver._w2) .= solver.c # backup the previous primal iterate

    F = get_F(solver.c,solver.f,solver.zl,solver.zu,solver.jacl,solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,solver.mu)
    solver.cnt.t = 0
    solver.alpha_z = 0.
    solver.ftype = "R"

    while true
        solver.alpha = min(get_alpha_max(solver.x,solver.xl,solver.xu,primal(solver.d),solver.tau),
                        get_alpha_z(solver.zl_r,solver.zu_r,dual_lb(solver.d),dual_ub(solver.d),solver.tau))

        solver.x .+= solver.alpha.* primal(solver.d)
        solver.y .+= solver.alpha.* dual(solver.d)
        solver.zl_r.+=solver.alpha.* dual_lb(solver.d)
        solver.zu_r.+=solver.alpha.* dual_ub(solver.d)

        eval_cons_wrapper!(solver,solver.c,solver.x)
        eval_grad_f_wrapper!(solver,solver.f,solver.x)
        solver.obj_val = eval_f_wrapper(solver,solver.x)

        !solver.opt.jacobian_constant && eval_jac_wrapper!(solver,solver.kkt,solver.x)
        jtprod!(solver.jacl,solver.kkt,solver.y)

        F_trial = get_F(
            solver.c,solver.f,solver.zl,solver.zu,solver.jacl,solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,solver.mu)
        if F_trial > solver.opt.soft_resto_pderror_reduction_factor*F
            solver.x .= primal(solver._w1)
            solver.y .= dual(solver._w1)
            solver.c .= dual(solver._w2) # backup the previous primal iterate
            return ROBUST
        end

        adjusted = adjust_boundary!(solver.x_lr,solver.xl_r,solver.x_ur,solver.xu_r,solver.mu)
        adjusted > 0 &&
            @warn(solver.logger,"In iteration $(solver.cnt.k), $adjusted Slack too small, adjusting variable bound")


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
        solver.inf_du = get_inf_du(solver.f,solver.zl,solver.zu,solver.jacl,sd)

        solver.inf_compl = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,0.,sc)
        inf_compl_mu = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,solver.mu,sc)
        print_iter(solver)

        !solver.opt.hessian_constant && eval_lag_hess_wrapper!(solver,solver.kkt,solver.x,solver.y)
        set_aug_diagonal!(solver.kkt,solver)
        set_aug_rhs!(solver, solver.kkt, solver.c)

        dual_inf_perturbation!(primal(solver.p),solver.ind_llb,solver.ind_uub,solver.mu,solver.opt.kappa_d)
        factorize_wrapper!(solver)
        solve_refine_wrapper!(solver,solver.d,solver.p)
        finish_aug_solve!(solver, solver.kkt, solver.mu)

        solver.ftype = "f"
    end
end

function robust!(solver::MadNLPSolver)
    initialize_robust_restorer!(solver)
    RR = solver.RR
    while true
        if !solver.opt.jacobian_constant
            eval_jac_wrapper!(solver, solver.kkt, solver.x)
        end
        jtprod!(solver.jacl, solver.kkt, solver.y)
        fixed_variable_treatment_vec!(solver.jacl,solver.ind_fixed)
        fixed_variable_treatment_z!(solver.zl,solver.zu,solver.f,solver.jacl,solver.ind_fixed)

        # evaluate termination criteria
        @trace(solver.logger,"Evaluating restoration phase termination criteria.")
        sd = get_sd(solver.y,solver.zl_r,solver.zu_r,solver.opt.s_max)
        sc = get_sc(solver.zl_r,solver.zu_r,solver.opt.s_max)
        solver.inf_pr = get_inf_pr(solver.c)
        solver.inf_du = get_inf_du(solver.f,solver.zl,solver.zu,solver.jacl,sd)
        solver.inf_compl = get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,0.,sc)

        # Robust restoration phase error
        RR.inf_pr_R = get_inf_pr_R(solver.c,RR.pp,RR.nn)
        RR.inf_du_R = get_inf_du_R(RR.f_R,solver.y,solver.zl,solver.zu,solver.jacl,RR.zp,RR.zn,solver.opt.rho,sd)
        RR.inf_compl_R = get_inf_compl_R(
            solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,0.,sc)
        inf_compl_mu_R = get_inf_compl_R(
            solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,sc)

        print_iter(solver;is_resto=true)

        max(RR.inf_pr_R,RR.inf_du_R,RR.inf_compl_R) <= solver.opt.tol && return INFEASIBLE_PROBLEM_DETECTED
        solver.cnt.k>=solver.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED


        # update the barrier parameter
        @trace(solver.logger,"Updating restoration phase barrier parameter.")
        while RR.mu_R != solver.opt.mu_min*100 &&
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
        set_aug_rhs_RR!(solver, solver.kkt, RR, solver.opt.rho)

        # without inertia correction,
        @trace(solver.logger,"Solving restoration phase primal-dual system.")
        factorize_wrapper!(solver)
        solve_refine_wrapper!(solver,solver.d,solver.p)

        finish_aug_solve!(solver, solver.kkt, RR.mu_R)
        finish_aug_solve_RR!(RR.dpp,RR.dnn,RR.dzp,RR.dzn,solver.y,dual(solver.d),RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,solver.opt.rho)


        theta_R = get_theta_R(solver.c,RR.pp,RR.nn)
        varphi_R = get_varphi_R(RR.obj_val_R,solver.x_lr,solver.xl_r,solver.xu_r,solver.x_ur,RR.pp,RR.nn,RR.mu_R)
        varphi_d_R = get_varphi_d_R(RR.f_R,solver.x,solver.xl,solver.xu,primal(solver.d),RR.pp,RR.nn,RR.dpp,RR.dnn,RR.mu_R,solver.opt.rho)

        # set alpha_min
        alpha_max = get_alpha_max_R(solver.x,solver.xl,solver.xu,primal(solver.d),RR.pp,RR.dpp,RR.nn,RR.dnn,RR.tau_R)

        
        solver.alpha_z = get_alpha_z_R(solver.zl_r,solver.zu_r,dual_lb(solver.d),dual_ub(solver.d),RR.zp,RR.dzp,RR.zn,RR.dzn,RR.tau_R)
        alpha_min = get_alpha_min(theta_R,varphi_d_R,solver.theta_min,solver.opt.gamma_theta,solver.opt.gamma_phi,
                                  solver.opt.alpha_min_frac,solver.opt.delta,solver.opt.s_theta,solver.opt.s_phi)

        # println(alpha_max)
        # println(solver.alpha_z)
        # filter start
        @trace(solver.logger,"Backtracking line search initiated.")
        solver.alpha = alpha_max
        solver.cnt.l = 1
        theta_R_trial = 0.
        varphi_R_trial = 0.
        small_search_norm = get_rel_search_norm(solver.x, primal(solver.d)) < 10*eps(eltype(solver.x))
        switching_condition = is_switching(varphi_d_R,solver.alpha,solver.opt.s_phi,solver.opt.delta,theta_R,solver.opt.s_theta)
        armijo_condition = false

        while true
            copyto!(solver.x_trial,solver.x)
            copyto!(RR.pp_trial,RR.pp)
            copyto!(RR.nn_trial,RR.nn)
            axpy!(solver.alpha,primal(solver.d),solver.x_trial)
            axpy!(solver.alpha,RR.dpp,RR.pp_trial)
            axpy!(solver.alpha,RR.dnn,RR.nn_trial)

            RR.obj_val_R_trial = get_obj_val_R(
                RR.pp_trial,RR.nn_trial,RR.D_R,solver.x_trial,RR.x_ref,solver.opt.rho,RR.zeta)
            eval_cons_wrapper!(solver, solver.c_trial, solver.x_trial)
            theta_R_trial  = get_theta_R(solver.c_trial,RR.pp_trial,RR.nn_trial)
            varphi_R_trial = get_varphi_R(
                RR.obj_val_R_trial,solver.x_trial_lr,solver.xl_r,solver.xu_r,solver.x_trial_ur,RR.pp_trial,RR.nn_trial,RR.mu_R)

            armijo_condition = is_armijo(varphi_R_trial,varphi_R,0.,solver.alpha,varphi_d_R) #####

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
                return RESTORATION_FAILED
            else
                @trace(solver.logger,"Step rejected; proceed with the next trial step.")
                solver.alpha < eps(eltype(solver.x))*10 && return solver.cnt.acceptable_cnt >0 ?
                    SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
            end
        end

        @trace(solver.logger,"Updating primal-dual variables.")
        solver.x.=solver.x_trial
        solver.c.=solver.c_trial
        RR.pp.=RR.pp_trial
        RR.nn.=RR.nn_trial

        RR.obj_val_R=RR.obj_val_R_trial
        RR.f_R .= RR.zeta.*RR.D_R.^2 .*(solver.x.-RR.x_ref)

        axpy!(solver.alpha, dual(solver.d), solver.y)
        axpy!(solver.alpha_z, dual_lb(solver.d),solver.zl_r)
        axpy!(solver.alpha_z, dual_ub(solver.d),solver.zu_r)
        axpy!(solver.alpha_z, RR.dzp,RR.zp)
        axpy!(solver.alpha_z, RR.dzn,RR.zn)

        reset_bound_dual!(solver.zl,solver.x,solver.xl,RR.mu_R,solver.opt.kappa_sigma)
        reset_bound_dual!(solver.zu,solver.xu,solver.x,RR.mu_R,solver.opt.kappa_sigma)
        reset_bound_dual!(RR.zp,RR.pp,RR.mu_R,solver.opt.kappa_sigma)
        reset_bound_dual!(RR.zn,RR.nn,RR.mu_R,solver.opt.kappa_sigma)

        adjusted = adjust_boundary!(solver.x_lr,solver.xl_r,solver.x_ur,solver.xu_r,solver.mu)
        adjusted > 0 &&
            @warn(solver.logger,"In iteration $(solver.cnt.k), $adjusted Slack too small, adjusting variable bound")

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
            solver.zl_r.=1
            solver.zu_r.=1

            set_initial_rhs!(solver, solver.kkt)
            initialize!(solver.kkt)

            factorize_wrapper!(solver)
            solve_refine_wrapper!(solver,solver.d,solver.p)
            if norm(dual(solver.d), Inf)>solver.opt.constr_mult_init_max
                fill!(solver.y, 0.0)
            else
                copyto!(solver.y, dual(solver.d))
            end
            solver.cnt.k+=1

            return REGULAR
        end

        solver.cnt.k>=solver.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        @trace(solver.logger,"Proceeding to the next restoration phase iteration.")
        solver.cnt.k+=1
        solver.cnt.t+=1
    end
end

function inertia_based_reg(solver::MadNLPSolver)
    @trace(solver.logger,"Inertia-based regularization started.")

    factorize_wrapper!(solver)
    num_pos,num_zero,num_neg = inertia(solver.linear_solver)
    solve_status = num_zero!= 0 ? false : solve_refine_wrapper!(solver,solver.d,solver.p)

    n_trial = 0
    solver.del_w = del_w_prev = 0.0
    while !is_inertia_correct(solver.kkt, num_pos, num_zero, num_neg) || !solve_status
        @debug(solver.logger,"Primal-dual perturbed.")
        if solver.del_w == 0.0
            solver.del_w = solver.del_w_last==0. ? solver.opt.first_hessian_perturbation :
                max(solver.opt.min_hessian_perturbation,solver.opt.perturb_dec_fact*solver.del_w_last)
        else
            solver.del_w*= solver.del_w_last==0. ? solver.opt.perturb_inc_fact_first : solver.opt.perturb_inc_fact
            if solver.del_w>solver.opt.max_hessian_perturbation solver.cnt.k+=1
                @debug(solver.logger,"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        solver.del_c = (num_zero == 0 || !solve_status) ?
            solver.opt.jacobian_regularization_value * solver.mu^(solver.opt.jacobian_regularization_exponent) : 0.
        regularize_diagonal!(solver.kkt, solver.del_w - del_w_prev, solver.del_c)
        del_w_prev = solver.del_w

        factorize_wrapper!(solver)
        num_pos,num_zero,num_neg = inertia(solver.linear_solver)
        solve_status = num_zero!= 0 ? false : solve_refine_wrapper!(solver,solver.d,solver.p)
        n_trial += 1
    end
    solver.del_w != 0 && (solver.del_w_last = solver.del_w)

    return true
end

function inertia_free_reg(solver::MadNLPSolver)

    @trace(solver.logger,"Inertia-free regularization started.")
    p0 = solver._w1
    d0 = solver._w2
    t = primal(solver._w3)
    n = primal(solver._w2)
    wx= primal(solver._w4)
    fill!(dual(solver._w3), 0)

    g = solver.x_trial # just to avoid new allocation
    g .= solver.f.-solver.mu./(solver.x.-solver.xl).+solver.mu./(solver.xu.-solver.x).+solver.jacl

    fixed_variable_treatment_vec!(primal(solver._w1), solver.ind_fixed)
    fixed_variable_treatment_vec!(primal(solver.p),   solver.ind_fixed)
    fixed_variable_treatment_vec!(g, solver.ind_fixed)

    factorize_wrapper!(solver)
    solve_status = (solve_refine_wrapper!(solver,d0,p0) && solve_refine_wrapper!(solver,solver.d,solver.p))
    t .= primal(solver.d) .- n
    mul!(solver._w4, solver.kkt, solver._w3) # prepartation for curv_test
    n_trial = 0
    solver.del_w = del_w_prev = 0.

    while !curv_test(t,n,g,wx,solver.opt.inertia_free_tol)  || !solve_status
        @debug(solver.logger,"Primal-dual perturbed.")
        if n_trial == 0
            solver.del_w = solver.del_w_last==.0 ? solver.opt.first_hessian_perturbation :
                max(solver.opt.min_hessian_perturbation,solver.opt.perturb_dec_fact*solver.del_w_last)
        else
            solver.del_w*= solver.del_w_last==.0 ? solver.opt.perturb_inc_fact_first : solver.opt.perturb_inc_fact
            if solver.del_w>solver.opt.max_hessian_perturbation solver.cnt.k+=1
                @debug(solver.logger,"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        solver.del_c = !solve_status ?
            solver.opt.jacobian_regularization_value * solver.mu^(solver.opt.jacobian_regularization_exponent) : 0.
        regularize_diagonal!(solver.kkt, solver.del_w - del_w_prev, solver.del_c)
        del_w_prev = solver.del_w

        factorize_wrapper!(solver)
        solve_status = (solve_refine_wrapper!(solver,d0,p0) && solve_refine_wrapper!(solver,solver.d,solver.p))
        t .= primal(solver.d) .- n
        mul!(solver._w4, solver.kkt, solver._w3) # prepartation for curv_test
        n_trial += 1
    end

    solver.del_w != 0 && (solver.del_w_last = solver.del_w)
    return true
end

curv_test(t,n,g,wx,inertia_free_tol) = dot(wx,t) + max(dot(wx,n)-dot(g,n),0) - inertia_free_tol*dot(t,t) >=0

function second_order_correction(solver::AbstractMadNLPSolver,alpha_max,theta,varphi,
                                 theta_trial,varphi_d,switching_condition::Bool)
    @trace(solver.logger,"Second-order correction started.")

    dual(solver._w1) .= alpha_max .* solver.c .+ solver.c_trial
    theta_soc_old = theta_trial
    for p=1:solver.opt.max_soc
        # compute second order correction
        set_aug_rhs!(solver, solver.kkt, dual(solver._w1))
        dual_inf_perturbation!(primal(solver.p),solver.ind_llb,solver.ind_uub,solver.mu,solver.opt.kappa_d)
        solve_refine_wrapper!(solver,solver._w1,solver.p)
        alpha_soc = get_alpha_max(solver.x,solver.xl,solver.xu,primal(solver._w1),solver.tau)

        solver.x_trial .= solver.x .+ alpha_soc .* primal(solver._w1)
        eval_cons_wrapper!(solver, solver.c_trial,solver.x_trial)
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


