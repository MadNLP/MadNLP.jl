
#=
    Regular filter line-search algorithm.
=#

function filter_line_search!(solver::AbstractMadNLPSolver{T}) where T
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

    if !switching_condition || !armijo_condition
        @trace(solver.logger,"Augmenting filter.")
        augment_filter!(solver.filter,theta_trial,varphi_trial,solver.opt.gamma_theta)
    end

    return LINESEARCH_SUCCEEDED
end

#=
    Variant of filter line-search for feasibility restoration phase.
=#

function filter_line_search_RR!(solver::AbstractMadNLPSolver{T}) where T
    RR = solver.RR
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

    if !switching_condition || !armijo_condition
        @trace(solver.logger,"Augmenting restoration phase filter.")
        augment_filter!(RR.filter,theta_R_trial,varphi_R_trial,solver.opt.gamma_theta)
    end

    return LINESEARCH_SUCCEEDED
end
