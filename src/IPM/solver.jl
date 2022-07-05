"""
TODO
"""
function initialize!(ips::AbstractInteriorPointSolver)
    # initializing slack variables
    @trace(ips.logger,"Initializing slack variables.")
    cons!(ips.nlp,get_x0(ips.nlp),_madnlp_unsafe_wrap(ips.c,get_ncon(ips.nlp)))
    ips.cnt.con_cnt += 1
    ips.x_slk.=ips.c_slk

    # Initialization
    @trace(ips.logger,"Initializing primal and bound duals.")
    ips.zl_r.=1.0
    ips.zu_r.=1.0
    ips.xl_r.-= max.(1,abs.(ips.xl_r)).*ips.opt.tol
    ips.xu_r.+= max.(1,abs.(ips.xu_r)).*ips.opt.tol
    initialize_variables!(ips.x,ips.xl,ips.xu,ips.opt.bound_push,ips.opt.bound_fac)

    # Automatic scaling (constraints)
    @trace(ips.logger,"Computing constraint scaling.")
    eval_jac_wrapper!(ips, ips.kkt, ips.x)
    compress_jacobian!(ips.kkt)
    if (ips.m > 0) && ips.opt.nlp_scaling
        jac = get_raw_jacobian(ips.kkt)
        scale_constraints!(ips.nlp, ips.con_scale, jac; max_gradient=ips.opt.nlp_scaling_max_gradient)
        set_jacobian_scaling!(ips.kkt, ips.con_scale)
        ips.l./=ips.con_scale
    end
    compress_jacobian!(ips.kkt)

    # Automatic scaling (objective)
    eval_grad_f_wrapper!(ips, ips.f,ips.x)
    @trace(ips.logger,"Computing objective scaling.")
    if ips.opt.nlp_scaling
        ips.obj_scale[] = scale_objective(ips.nlp, ips.f; max_gradient=ips.opt.nlp_scaling_max_gradient)
        ips.f.*=ips.obj_scale[]
    end

    # Initialize dual variables
    @trace(ips.logger,"Initializing constraint duals.")
    if !ips.opt.dual_initialized
        set_initial_rhs!(ips, ips.kkt)
        initialize!(ips.kkt)
        factorize_wrapper!(ips)
        solve_refine_wrapper!(ips,ips.d,ips.p)
        if norm(dual(ips.d), Inf) > ips.opt.constr_mult_init_max
            fill!(ips.l, 0.0)
        else
            copyto!(ips.l, dual(ips.d))
        end
    end

    # Initializing
    ips.obj_val = eval_f_wrapper(ips, ips.x)
    eval_cons_wrapper!(ips, ips.c, ips.x)
    eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.l)

    theta = get_theta(ips.c)
    ips.theta_max=1e4*max(1,theta)
    ips.theta_min=1e-4*max(1,theta)
    ips.mu=ips.opt.mu_init
    ips.tau=max(ips.opt.tau_min,1-ips.opt.mu_init)
    ips.filter = [(ips.theta_max,-Inf)]

    return REGULAR
end


"""
TODO
"""
function reinitialize!(ips::AbstractInteriorPointSolver)
    view(ips.x,1:get_nvar(ips.nlp)) .= get_x0(ips.nlp)

    ips.obj_val = eval_f_wrapper(ips, ips.x)
    eval_grad_f_wrapper!(ips, ips.f, ips.x)
    eval_cons_wrapper!(ips, ips.c, ips.x)
    eval_jac_wrapper!(ips, ips.kkt, ips.x)
    eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.l)

    theta = get_theta(ips.c)
    ips.theta_max=1e4*max(1,theta)
    ips.theta_min=1e-4*max(1,theta)
    ips.mu=ips.opt.mu_init
    ips.tau=max(ips.opt.tau_min,1-ips.opt.mu_init)
    ips.filter = [(ips.theta_max,-Inf)]

    return REGULAR
end

# major loops ---------------------------------------------------------
"""
TODO
"""
function optimize!(ips::AbstractInteriorPointSolver)
    try
        if ips.status == INITIAL
            @notice(ips.logger,"This is $(introduce()), running with $(introduce(ips.linear_solver))\n")
            print_init(ips)
            ips.status = initialize!(ips)
        else # resolving the problem
            ips.status = reinitialize!(ips)
        end

        while ips.status >= REGULAR
            ips.status == REGULAR && (ips.status = regular!(ips))
            ips.status == RESTORE && (ips.status = restore!(ips))
            ips.status == ROBUST && (ips.status = robust!(ips))
        end
    catch e
        if e isa InvalidNumberException
            ips.status=INVALID_NUMBER_DETECTED
        elseif e isa NotEnoughDegreesOfFreedomException
            ips.status=NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa LinearSolverException
            ips.status=ERROR_IN_STEP_COMPUTATION;
            ips.opt.rethrow_error && rethrow(e)
        elseif e isa InterruptException
            ips.status=USER_REQUESTED_STOP
            ips.opt.rethrow_error && rethrow(e)
        else
            ips.status=INTERNAL_ERROR
            ips.opt.rethrow_error && rethrow(e)
        end
    finally
        ips.cnt.total_time = time() - ips.cnt.start_time
        !(ips.status < SOLVE_SUCCEEDED) && (print_summary_1(ips);print_summary_2(ips))
        # Unscale once the summary has been printed out
        unscale!(ips)
        @notice(ips.logger,"EXIT: $(STATUS_OUTPUT_DICT[ips.status])")
        ips.opt.disable_garbage_collector &&
            (GC.enable(true); @warn(ips.logger,"Julia garbage collector is turned back on"))
        finalize(ips.logger)
    end
    return MadNLPExecutionStats(ips)
end

function unscale!(ips::AbstractInteriorPointSolver)
    ips.obj_val/=ips.obj_scale[]
    ips.c ./= ips.con_scale
    ips.c .-= ips.rhs
    ips.c_slk .+= ips.x_slk
end

"""
TODO
"""
function regular!(ips::AbstractInteriorPointSolver)
    while true
        if (ips.cnt.k!=0 && !ips.opt.jacobian_constant)
            eval_jac_wrapper!(ips, ips.kkt, ips.x)
        end
        jtprod!(ips.jacl, ips.kkt, ips.l)
        fixed_variable_treatment_vec!(ips.jacl,ips.ind_fixed)
        fixed_variable_treatment_z!(ips.zl,ips.zu,ips.f,ips.jacl,ips.ind_fixed)

        sd = get_sd(ips.l,ips.zl_r,ips.zu_r,ips.opt.s_max)
        sc = get_sc(ips.zl_r,ips.zu_r,ips.opt.s_max)

        ips.inf_pr = get_inf_pr(ips.c)
        ips.inf_du = get_inf_du(ips.f,ips.zl,ips.zu,ips.jacl,sd)
        ips.inf_compl = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,0.,sc)
        inf_compl_mu = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,ips.mu,sc)

        print_iter(ips)

        # evaluate termination criteria
        @trace(ips.logger,"Evaluating termination criteria.")
        max(ips.inf_pr,ips.inf_du,ips.inf_compl) <= ips.opt.tol && return SOLVE_SUCCEEDED
        max(ips.inf_pr,ips.inf_du,ips.inf_compl) <= ips.opt.acceptable_tol ?
            (ips.cnt.acceptable_cnt < ips.opt.acceptable_iter ?
            ips.cnt.acceptable_cnt+=1 : return SOLVED_TO_ACCEPTABLE_LEVEL) : (ips.cnt.acceptable_cnt = 0)
        max(ips.inf_pr,ips.inf_du,ips.inf_compl) >= ips.opt.diverging_iterates_tol && return DIVERGING_ITERATES
        ips.cnt.k>=ips.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-ips.cnt.start_time>=ips.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        @trace(ips.logger,"Updating the barrier parameter.")
        while ips.mu != max(ips.opt.mu_min,ips.opt.tol/10) &&
            max(ips.inf_pr,ips.inf_du,inf_compl_mu) <= ips.opt.barrier_tol_factor*ips.mu
            mu_new = get_mu(ips.mu,ips.opt.mu_min,
                            ips.opt.mu_linear_decrease_factor,ips.opt.mu_superlinear_decrease_power,ips.opt.tol)
            inf_compl_mu = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,ips.mu,sc)
            ips.tau= get_tau(ips.mu,ips.opt.tau_min)
            ips.mu = mu_new
            empty!(ips.filter)
            push!(ips.filter,(ips.theta_max,-Inf))
        end

        # compute the newton step
        @trace(ips.logger,"Computing the newton step.")
        if (ips.cnt.k!=0 && !ips.opt.hessian_constant)
            eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.l)
        end

        set_aug_diagonal!(ips.kkt,ips)
        set_aug_rhs!(ips, ips.kkt, ips.c)
        if ips.opt.inertia_correction_method == INERTIA_FREE
            set_aug_rhs_ifr!(ips, ips.kkt)
        end
        dual_inf_perturbation!(primal(ips.p),ips.ind_llb,ips.ind_uub,ips.mu,ips.opt.kappa_d)

        # start inertia conrrection
        @trace(ips.logger,"Solving primal-dual system.")
        if ips.opt.inertia_correction_method == INERTIA_FREE
            inertia_free_reg(ips) || return ROBUST
        elseif ips.opt.inertia_correction_method == INERTIA_BASED
            inertia_based_reg(ips) || return ROBUST
        end

        finish_aug_solve!(ips, ips.kkt, ips.mu)

        # filter start
        @trace(ips.logger,"Backtracking line search initiated.")
        theta = get_theta(ips.c)
        varphi= get_varphi(ips.obj_val,ips.x_lr,ips.xl_r,ips.xu_r,ips.x_ur,ips.mu)
        varphi_d = get_varphi_d(ips.f,ips.x,ips.xl,ips.xu,primal(ips.d),ips.mu)


        alpha_max = get_alpha_max(ips.x,ips.xl,ips.xu,primal(ips.d),ips.tau)
        ips.alpha_z = get_alpha_z(ips.zl_r,ips.zu_r,dual_lb(ips.d),dual_ub(ips.d),ips.tau)
        alpha_min = get_alpha_min(theta,varphi_d,ips.theta_min,ips.opt.gamma_theta,ips.opt.gamma_phi,
                                  ips.opt.alpha_min_frac,ips.opt.delta,ips.opt.s_theta,ips.opt.s_phi)
        ips.cnt.l = 1
        ips.alpha = alpha_max
        varphi_trial= 0.
            theta_trial = 0.
            small_search_norm = get_rel_search_norm(ips.x, primal(ips.d)) < 10*eps(eltype(ips.x))
        switching_condition = is_switching(varphi_d,ips.alpha,ips.opt.s_phi,ips.opt.delta,2.,ips.opt.s_theta)
        armijo_condition = false
        while true
            copyto!(ips.x_trial, ips.x)
            axpy!(ips.alpha, primal(ips.d), ips.x_trial)

            ips.obj_val_trial = eval_f_wrapper(ips, ips.x_trial)
            eval_cons_wrapper!(ips, ips.c_trial, ips.x_trial)

            theta_trial = get_theta(ips.c_trial)
            varphi_trial= get_varphi(ips.obj_val_trial,ips.x_trial_lr,ips.xl_r,ips.xu_r,ips.x_trial_ur,ips.mu)
            armijo_condition = is_armijo(varphi_trial,varphi,ips.opt.eta_phi,ips.alpha,varphi_d)

            small_search_norm && break

            ips.ftype = get_ftype(
                ips.filter,theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
                ips.theta_min,ips.opt.obj_max_inc,ips.opt.gamma_theta,ips.opt.gamma_phi,
                has_constraints(ips))
            ips.ftype in ["f","h"] && (@trace(ips.logger,"Step accepted with type $(ips.ftype)"); break)

            ips.cnt.l==1 && theta_trial>=theta && second_order_correction(
                ips,alpha_max,theta,varphi,theta_trial,varphi_d,switching_condition) && break

            ips.alpha /= 2
            ips.cnt.l += 1
            if ips.alpha < alpha_min
                @debug(ips.logger,
                       "Cannot find an acceptable step at iteration $(ips.cnt.k). Switching to restoration phase.")
                ips.cnt.k+=1
                return RESTORE
            else
                @trace(ips.logger,"Step rejected; proceed with the next trial step.")
                ips.alpha * norm(primal(ips.d)) < eps(eltype(ips.x))*10 &&
                    return ips.cnt.acceptable_cnt >0 ?
                    SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
            end
        end

        @trace(ips.logger,"Updating primal-dual variables.")
        ips.x.=ips.x_trial
        ips.c.=ips.c_trial
        ips.obj_val=ips.obj_val_trial
        adjusted = adjust_boundary!(ips.x_lr,ips.xl_r,ips.x_ur,ips.xu_r,ips.mu)
        adjusted > 0 &&
            @warn(ips.logger,"In iteration $(ips.cnt.k), $adjusted Slack too small, adjusting variable bound")

        axpy!(ips.alpha,dual(ips.d),ips.l)
        axpy!(ips.alpha_z, dual_lb(ips.d), ips.zl_r)
        axpy!(ips.alpha_z, dual_ub(ips.d), ips.zu_r)
        reset_bound_dual!(ips.zl,ips.x,ips.xl,ips.mu,ips.opt.kappa_sigma)
        reset_bound_dual!(ips.zu,ips.xu,ips.x,ips.mu,ips.opt.kappa_sigma)
        eval_grad_f_wrapper!(ips, ips.f,ips.x)

        if !switching_condition || !armijo_condition
            @trace(ips.logger,"Augmenting filter.")
            augment_filter!(ips.filter,theta_trial,varphi_trial,ips.opt.gamma_theta)
        end

        ips.cnt.k+=1
        @trace(ips.logger,"Proceeding to the next interior point iteration.")
    end
end

"""
TODO
"""
function restore!(ips::AbstractInteriorPointSolver)
    ips.del_w=0
    primal(ips._w1) .= ips.x # backup the previous primal iterate
    dual(ips._w1) .= ips.l # backup the previous primal iterate
    dual(ips._w2) .= ips.c # backup the previous primal iterate

    F = get_F(ips.c,ips.f,ips.zl,ips.zu,ips.jacl,ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,ips.mu)
    ips.cnt.t = 0
    ips.alpha_z = 0.
    ips.ftype = "R"

    while true
        ips.alpha = min(get_alpha_max(ips.x,ips.xl,ips.xu,primal(ips.d),ips.tau),
                        get_alpha_z(ips.zl_r,ips.zu_r,dual_lb(ips.d),dual_ub(ips.d),ips.tau))

        ips.x .+= ips.alpha.* primal(ips.d)
        ips.l .+= ips.alpha.* dual(ips.d)
        ips.zl_r.+=ips.alpha.* dual_lb(ips.d)
        ips.zu_r.+=ips.alpha.* dual_ub(ips.d)

        eval_cons_wrapper!(ips,ips.c,ips.x)
        eval_grad_f_wrapper!(ips,ips.f,ips.x)
        ips.obj_val = eval_f_wrapper(ips,ips.x)

        !ips.opt.jacobian_constant && eval_jac_wrapper!(ips,ips.kkt,ips.x)
        jtprod!(ips.jacl,ips.kkt,ips.l)

        F_trial = get_F(
            ips.c,ips.f,ips.zl,ips.zu,ips.jacl,ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,ips.mu)
        if F_trial > ips.opt.soft_resto_pderror_reduction_factor*F
            ips.x .= primal(ips._w1)
            ips.l .= dual(ips._w1)
            ips.c .= dual(ips._w2) # backup the previous primal iterate
            return ROBUST
        end

        adjusted = adjust_boundary!(ips.x_lr,ips.xl_r,ips.x_ur,ips.xu_r,ips.mu)
        adjusted > 0 &&
            @warn(ips.logger,"In iteration $(ips.cnt.k), $adjusted Slack too small, adjusting variable bound")


        F = F_trial

        theta = get_theta(ips.c)
        varphi= get_varphi(ips.obj_val,ips.x_lr,ips.xl_r,ips.xu_r,ips.x_ur,ips.mu)

        ips.cnt.k+=1

        is_filter_acceptable(ips.filter,theta,varphi) ? (return REGULAR) : (ips.cnt.t+=1)
        ips.cnt.k>=ips.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-ips.cnt.start_time>=ips.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED


        sd = get_sd(ips.l,ips.zl_r,ips.zu_r,ips.opt.s_max)
        sc = get_sc(ips.zl_r,ips.zu_r,ips.opt.s_max)
        ips.inf_pr = get_inf_pr(ips.c)
        ips.inf_du = get_inf_du(ips.f,ips.zl,ips.zu,ips.jacl,sd)

        ips.inf_compl = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,0.,sc)
        inf_compl_mu = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,ips.mu,sc)
        print_iter(ips)

        !ips.opt.hessian_constant && eval_lag_hess_wrapper!(ips,ips.kkt,ips.x,ips.l)
        set_aug_diagonal!(ips.kkt,ips)
        set_aug_rhs!(ips, ips.kkt, ips.c)

        dual_inf_perturbation!(primal(ips.p),ips.ind_llb,ips.ind_uub,ips.mu,ips.opt.kappa_d)
        factorize_wrapper!(ips)
        solve_refine_wrapper!(ips,ips.d,ips.p)
        finish_aug_solve!(ips, ips.kkt, ips.mu)

        ips.ftype = "f"
    end
end

"""
TODO
"""
function robust!(ips::InteriorPointSolver)
    initialize_robust_restorer!(ips)
    RR = ips.RR
    while true
        if !ips.opt.jacobian_constant
            eval_jac_wrapper!(ips, ips.kkt, ips.x)
        end
        jtprod!(ips.jacl, ips.kkt, ips.l)
        fixed_variable_treatment_vec!(ips.jacl,ips.ind_fixed)
        fixed_variable_treatment_z!(ips.zl,ips.zu,ips.f,ips.jacl,ips.ind_fixed)

        # evaluate termination criteria
        @trace(ips.logger,"Evaluating restoration phase termination criteria.")
        sd = get_sd(ips.l,ips.zl_r,ips.zu_r,ips.opt.s_max)
        sc = get_sc(ips.zl_r,ips.zu_r,ips.opt.s_max)
        ips.inf_pr = get_inf_pr(ips.c)
        ips.inf_du = get_inf_du(ips.f,ips.zl,ips.zu,ips.jacl,sd)
        ips.inf_compl = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,0.,sc)

        # Robust restoration phase error
        RR.inf_pr_R = get_inf_pr_R(ips.c,RR.pp,RR.nn)
        RR.inf_du_R = get_inf_du_R(RR.f_R,ips.l,ips.zl,ips.zu,ips.jacl,RR.zp,RR.zn,ips.opt.rho,sd)
        RR.inf_compl_R = get_inf_compl_R(
            ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,0.,sc)
        inf_compl_mu_R = get_inf_compl_R(
            ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,sc)

        print_iter(ips;is_resto=true)

        max(RR.inf_pr_R,RR.inf_du_R,RR.inf_compl_R) <= ips.opt.tol && return INFEASIBLE_PROBLEM_DETECTED
        ips.cnt.k>=ips.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-ips.cnt.start_time>=ips.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED


        # update the barrier parameter
        @trace(ips.logger,"Updating restoration phase barrier parameter.")
        while RR.mu_R != ips.opt.mu_min*100 &&
            max(RR.inf_pr_R,RR.inf_du_R,inf_compl_mu_R) <= ips.opt.barrier_tol_factor*RR.mu_R
            RR.mu_R = get_mu(RR.mu_R,ips.opt.mu_min,
                            ips.opt.mu_linear_decrease_factor,ips.opt.mu_superlinear_decrease_power,ips.opt.tol)
            inf_compl_mu_R = get_inf_compl_R(
                ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,sc)
            RR.tau_R= max(ips.opt.tau_min,1-RR.mu_R)
            RR.zeta = sqrt(RR.mu_R)

            empty!(RR.filter)
            push!(RR.filter,(ips.theta_max,-Inf))
        end

        # compute the newton step
        if !ips.opt.hessian_constant
            eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.l; is_resto=true)
        end
        set_aug_RR!(ips.kkt, ips, RR)
        set_aug_rhs_RR!(ips, ips.kkt, RR, ips.opt.rho)

        # without inertia correction,
        @trace(ips.logger,"Solving restoration phase primal-dual system.")
        factorize_wrapper!(ips)
        solve_refine_wrapper!(ips,ips.d,ips.p)

        finish_aug_solve!(ips, ips.kkt, RR.mu_R)
        finish_aug_solve_RR!(RR.dpp,RR.dnn,RR.dzp,RR.dzn,ips.l,dual(ips.d),RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,ips.opt.rho)


        theta_R = get_theta_R(ips.c,RR.pp,RR.nn)
        varphi_R = get_varphi_R(RR.obj_val_R,ips.x_lr,ips.xl_r,ips.xu_r,ips.x_ur,RR.pp,RR.nn,RR.mu_R)
        varphi_d_R = get_varphi_d_R(RR.f_R,ips.x,ips.xl,ips.xu,primal(ips.d),RR.pp,RR.nn,RR.dpp,RR.dnn,RR.mu_R,ips.opt.rho)

        # set alpha_min
        alpha_max = get_alpha_max_R(ips.x,ips.xl,ips.xu,primal(ips.d),RR.pp,RR.dpp,RR.nn,RR.dnn,RR.tau_R)
        ips.alpha_z = get_alpha_z_R(ips.zl_r,ips.zu_r,dual_lb(ips.d),dual_ub(ips.d),RR.zp,RR.dzp,RR.zn,RR.dzn,RR.tau_R)
        alpha_min = get_alpha_min(theta_R,varphi_d_R,ips.theta_min,ips.opt.gamma_theta,ips.opt.gamma_phi,
                                  ips.opt.alpha_min_frac,ips.opt.delta,ips.opt.s_theta,ips.opt.s_phi)

        # filter start
        @trace(ips.logger,"Backtracking line search initiated.")
        ips.alpha = alpha_max
        ips.cnt.l = 1
        theta_R_trial = 0.
        varphi_R_trial = 0.
        small_search_norm = get_rel_search_norm(ips.x, primal(ips.d)) < 10*eps(eltype(ips.x))
        switching_condition = is_switching(varphi_d_R,ips.alpha,ips.opt.s_phi,ips.opt.delta,theta_R,ips.opt.s_theta)
        armijo_condition = false

        while true
            copyto!(ips.x_trial,ips.x)
            copyto!(RR.pp_trial,RR.pp)
            copyto!(RR.nn_trial,RR.nn)
            axpy!(ips.alpha,primal(ips.d),ips.x_trial)
            axpy!(ips.alpha,RR.dpp,RR.pp_trial)
            axpy!(ips.alpha,RR.dnn,RR.nn_trial)

            RR.obj_val_R_trial = get_obj_val_R(
                RR.pp_trial,RR.nn_trial,RR.D_R,ips.x_trial,RR.x_ref,ips.opt.rho,RR.zeta)
            eval_cons_wrapper!(ips, ips.c_trial, ips.x_trial)
            theta_R_trial  = get_theta_R(ips.c_trial,RR.pp_trial,RR.nn_trial)
            varphi_R_trial = get_varphi_R(
                RR.obj_val_R_trial,ips.x_trial_lr,ips.xl_r,ips.xu_r,ips.x_trial_ur,RR.pp_trial,RR.nn_trial,RR.mu_R)

            armijo_condition = is_armijo(varphi_R_trial,varphi_R,0.,ips.alpha,varphi_d_R) #####

            small_search_norm && break
            ips.ftype = get_ftype(
                RR.filter,theta_R,theta_R_trial,varphi_R,varphi_R_trial,
                switching_condition,armijo_condition,
                ips.theta_min,ips.opt.obj_max_inc,ips.opt.gamma_theta,ips.opt.gamma_phi,
                has_constraints(ips))
            ips.ftype in ["f","h"] && (@trace(ips.logger,"Step accepted with type $(ips.ftype)"); break)

            ips.alpha /= 2
            ips.cnt.l += 1
            if ips.alpha < alpha_min
                @debug(ips.logger,"Restoration phase cannot find an acceptable step at iteration $(ips.cnt.k).")
                return RESTORATION_FAILED
            else
                @trace(ips.logger,"Step rejected; proceed with the next trial step.")
                ips.alpha < eps(eltype(ips.x))*10 && return ips.cnt.acceptable_cnt >0 ?
                    SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
            end
        end

        @trace(ips.logger,"Updating primal-dual variables.")
        ips.x.=ips.x_trial
        ips.c.=ips.c_trial
        RR.pp.=RR.pp_trial
        RR.nn.=RR.nn_trial

        RR.obj_val_R=RR.obj_val_R_trial
        RR.f_R .= RR.zeta.*RR.D_R.^2 .*(ips.x.-RR.x_ref)

        axpy!(ips.alpha, dual(ips.d), ips.l)
        axpy!(ips.alpha_z, dual_lb(ips.d),ips.zl_r)
        axpy!(ips.alpha_z, dual_ub(ips.d),ips.zu_r)
        axpy!(ips.alpha_z, RR.dzp,RR.zp)
        axpy!(ips.alpha_z, RR.dzn,RR.zn)

        reset_bound_dual!(ips.zl,ips.x,ips.xl,RR.mu_R,ips.opt.kappa_sigma)
        reset_bound_dual!(ips.zu,ips.xu,ips.x,RR.mu_R,ips.opt.kappa_sigma)
        reset_bound_dual!(RR.zp,RR.pp,RR.mu_R,ips.opt.kappa_sigma)
        reset_bound_dual!(RR.zn,RR.nn,RR.mu_R,ips.opt.kappa_sigma)

        adjusted = adjust_boundary!(ips.x_lr,ips.xl_r,ips.x_ur,ips.xu_r,ips.mu)
        adjusted > 0 &&
            @warn(ips.logger,"In iteration $(ips.cnt.k), $adjusted Slack too small, adjusting variable bound")

        if !switching_condition || !armijo_condition
            @trace(ips.logger,"Augmenting restoration phase filter.")
            augment_filter!(RR.filter,theta_R_trial,varphi_R_trial,ips.opt.gamma_theta)
        end

        # check if going back to regular phase
        @trace(ips.logger,"Checking if going back to regular phase.")
        ips.obj_val = eval_f_wrapper(ips, ips.x)
        eval_grad_f_wrapper!(ips, ips.f, ips.x)
        theta = get_theta(ips.c)
        varphi= get_varphi(ips.obj_val,ips.x_lr,ips.xl_r,ips.xu_r,ips.x_ur,ips.mu)

        if is_filter_acceptable(ips.filter,theta,varphi) &&
            theta <= ips.opt.required_infeasibility_reduction * RR.theta_ref

            @trace(ips.logger,"Going back to the regular phase.")
            ips.zl_r.=1
            ips.zu_r.=1

            set_initial_rhs!(ips, ips.kkt)
            initialize!(ips.kkt)

            factorize_wrapper!(ips)
            solve_refine_wrapper!(ips,ips.d,ips.p)
            if norm(dual(ips.d), Inf)>ips.opt.constr_mult_init_max
                fill!(ips.l, 0.0)
            else
                copyto!(ips.l, dual(ips.d))
            end
            ips.cnt.k+=1

            return REGULAR
        end

        ips.cnt.k>=ips.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-ips.cnt.start_time>=ips.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        @trace(ips.logger,"Proceeding to the next restoration phase iteration.")
        ips.cnt.k+=1
        ips.cnt.t+=1
    end
end

function inertia_based_reg(ips::InteriorPointSolver)
    @trace(ips.logger,"Inertia-based regularization started.")

    factorize_wrapper!(ips)
    num_pos,num_zero,num_neg = inertia(ips.linear_solver)
    solve_status = num_zero!= 0 ? false : solve_refine_wrapper!(ips,ips.d,ips.p)

    n_trial = 0
    ips.del_w = del_w_prev = 0.0
    while !is_inertia_correct(ips.kkt, num_pos, num_zero, num_neg) || !solve_status
        @debug(ips.logger,"Primal-dual perturbed.")
        if ips.del_w == 0.0
            ips.del_w = ips.del_w_last==0. ? ips.opt.first_hessian_perturbation :
                max(ips.opt.min_hessian_perturbation,ips.opt.perturb_dec_fact*ips.del_w_last)
        else
            ips.del_w*= ips.del_w_last==0. ? ips.opt.perturb_inc_fact_first : ips.opt.perturb_inc_fact
            if ips.del_w>ips.opt.max_hessian_perturbation ips.cnt.k+=1
                @debug(ips.logger,"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        ips.del_c = (num_zero == 0 || !solve_status) ?
            ips.opt.jacobian_regularization_value * ips.mu^(ips.opt.jacobian_regularization_exponent) : 0.
        regularize_diagonal!(ips.kkt, ips.del_w - del_w_prev, ips.del_c)
        del_w_prev = ips.del_w

        factorize_wrapper!(ips)
        num_pos,num_zero,num_neg = inertia(ips.linear_solver)
        solve_status = num_zero!= 0 ? false : solve_refine_wrapper!(ips,ips.d,ips.p)
        n_trial += 1
    end
    ips.del_w != 0 && (ips.del_w_last = ips.del_w)

    return true
end

function inertia_free_reg(ips::InteriorPointSolver)

    @trace(ips.logger,"Inertia-free regularization started.")
    p0 = ips._w1
    d0 = ips._w2
    t = primal(ips._w3)
    n = primal(ips._w2)
    wx= primal(ips._w4)
    fill!(dual(ips._w3), 0)

    g = ips.x_trial # just to avoid new allocation
    g .= ips.f.-ips.mu./(ips.x.-ips.xl).+ips.mu./(ips.xu.-ips.x).+ips.jacl

    fixed_variable_treatment_vec!(primal(ips._w1), ips.ind_fixed)
    fixed_variable_treatment_vec!(primal(ips.p),   ips.ind_fixed)
    fixed_variable_treatment_vec!(g, ips.ind_fixed)

    factorize_wrapper!(ips)
    solve_status = (solve_refine_wrapper!(ips,d0,p0) && solve_refine_wrapper!(ips,ips.d,ips.p))
    t .= primal(ips.d) .- n
    mul!(ips._w4, ips.kkt, ips._w3) # prepartation for curv_test
    n_trial = 0
    ips.del_w = del_w_prev = 0.

    while !curv_test(t,n,g,wx,ips.opt.inertia_free_tol)  || !solve_status
        @debug(ips.logger,"Primal-dual perturbed.")
        if n_trial == 0
            ips.del_w = ips.del_w_last==.0 ? ips.opt.first_hessian_perturbation :
                max(ips.opt.min_hessian_perturbation,ips.opt.perturb_dec_fact*ips.del_w_last)
        else
            ips.del_w*= ips.del_w_last==.0 ? ips.opt.perturb_inc_fact_first : ips.opt.perturb_inc_fact
            if ips.del_w>ips.opt.max_hessian_perturbation ips.cnt.k+=1
                @debug(ips.logger,"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        ips.del_c = !solve_status ?
            ips.opt.jacobian_regularization_value * ips.mu^(ips.opt.jacobian_regularization_exponent) : 0.
        regularize_diagonal!(ips.kkt, ips.del_w - del_w_prev, ips.del_c)
        del_w_prev = ips.del_w

        factorize_wrapper!(ips)
        solve_status = (solve_refine_wrapper!(ips,d0,p0) && solve_refine_wrapper!(ips,ips.d,ips.p))
        t .= primal(ips.d) .- n
        mul!(ips._w4, ips.kkt, ips._w3) # prepartation for curv_test
        n_trial += 1
    end

    ips.del_w != 0 && (ips.del_w_last = ips.del_w)
    return true
end

curv_test(t,n,g,wx,inertia_free_tol) = dot(wx,t) + max(dot(wx,n)-dot(g,n),0) - inertia_free_tol*dot(t,t) >=0

function second_order_correction(ips::AbstractInteriorPointSolver,alpha_max,theta,varphi,
                                 theta_trial,varphi_d,switching_condition::Bool)
    @trace(ips.logger,"Second-order correction started.")

    dual(ips._w1) .= alpha_max .* ips.c .+ ips.c_trial
    theta_soc_old = theta_trial
    for p=1:ips.opt.max_soc
        # compute second order correction
        set_aug_rhs!(ips, ips.kkt, dual(ips._w1))
        dual_inf_perturbation!(primal(ips.p),ips.ind_llb,ips.ind_uub,ips.mu,ips.opt.kappa_d)
        solve_refine_wrapper!(ips,ips._w1,ips.p)
        alpha_soc = get_alpha_max(ips.x,ips.xl,ips.xu,primal(ips._w1),ips.tau)

        ips.x_trial .= ips.x .+ alpha_soc .* primal(ips._w1)
        eval_cons_wrapper!(ips, ips.c_trial,ips.x_trial)
        ips.obj_val_trial = eval_f_wrapper(ips, ips.x_trial)

        theta_soc = get_theta(ips.c_trial)
        varphi_soc= get_varphi(ips.obj_val_trial,ips.x_trial_lr,ips.xl_r,ips.xu_r,ips.x_trial_ur,ips.mu)

        !is_filter_acceptable(ips.filter,theta_soc,varphi_soc) && break

        if theta <=ips.theta_min && switching_condition
            # Case I
            if is_armijo(varphi_soc,varphi,ips.opt.eta_phi,ips.alpha,varphi_d)
                @trace(ips.logger,"Step in second order correction accepted by armijo condition.")
                ips.ftype = "F"
                ips.alpha=alpha_soc
                return true
            end
        else
            # Case II
            if is_sufficient_progress(theta_soc,theta,ips.opt.gamma_theta,varphi_soc,varphi,ips.opt.gamma_phi,has_constraints(ips))
                @trace(ips.logger,"Step in second order correction accepted by sufficient progress.")
                ips.ftype = "H"
                ips.alpha=alpha_soc
                return true
            end
        end

        theta_soc>ips.opt.kappa_soc*theta_soc_old && break
        theta_soc_old = theta_soc
    end
    @trace(ips.logger,"Second-order correction terminated.")

    return false
end

