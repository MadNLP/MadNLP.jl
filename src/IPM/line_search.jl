
#=
    Regular filter line-search algorithm.
=#

function filter_line_search!(solver::AbstractMadNLPSolver{T}) where T
    theta = get_theta(_c(solver))
    varphi= get_varphi(_obj_val(solver),_x_lr(solver),_xl_r(solver),_xu_r(solver),_x_ur(solver),_mu(solver))
    varphi_d = get_varphi_d(
        primal(_f(solver)),
        primal(_x(solver)),
        primal(_xl(solver)),
        primal(_xu(solver)),
        primal(_d(solver)),
        _mu(solver),
    )

    alpha_max = get_alpha_max(
        primal(_x(solver)),
        primal(_xl(solver)),
        primal(_xu(solver)),
        primal(_d(solver)),
        _tau(solver),
    )
    set_alpha_z!(solver, get_alpha_z(_zl_r(solver),_zu_r(solver),dual_lb(_d(solver)),dual_ub(_d(solver)),_tau(solver)))
    alpha_min = get_alpha_min(theta,varphi_d,_theta_min(solver),_opt(solver).gamma_theta,_opt(solver).gamma_phi,
                                _opt(solver).alpha_min_frac,_opt(solver).delta,_opt(solver).s_theta,_opt(solver).s_phi)
    _cnt(solver).l = 1
    set_alpha!(solver, alpha_max)
    varphi_trial= zero(T)
    theta_trial = zero(T)
    small_search_norm = get_rel_search_norm(primal(_x(solver)), primal(_d(solver))) < 10*eps(T)
    switching_condition = is_switching(varphi_d,_alpha(solver),_opt(solver).s_phi,_opt(solver).delta,2.,_opt(solver).s_theta)
    armijo_condition = false
    unsuccessful_iterate = false

    while true

        copyto!(full(_x_trial(solver)), full(_x(solver)))
        axpy!(_alpha(solver), primal(_d(solver)), primal(_x_trial(solver)))
        set_obj_val_trial!(solver, eval_f_wrapper(solver, _x_trial(solver)))
        eval_cons_wrapper!(solver, _c_trial(solver), _x_trial(solver))

        theta_trial = get_theta(_c_trial(solver))
        varphi_trial= get_varphi(_obj_val_trial(solver),_x_trial_lr(solver),_xl_r(solver),_xu_r(solver),_x_trial_ur(solver),_mu(solver))
        armijo_condition = is_armijo(varphi_trial,varphi,_opt(solver).eta_phi,_alpha(solver),varphi_d)

        small_search_norm && break

        set_ftype!(solver, get_ftype(
            _filter(solver),theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
            _theta_min(solver),_opt(solver).obj_max_inc,_opt(solver).gamma_theta,_opt(solver).gamma_phi,
            has_constraints(solver))
                )
        if _ftype(solver) in ["f","h"]
            @trace(_logger(solver),"Step accepted with type $(_ftype(solver))")
            break
        end

        if _cnt(solver).l==1 && theta_trial>=theta
            if second_order_correction(
                solver,alpha_max,theta,varphi,theta_trial,varphi_d,switching_condition
            )
                break
            end
        end

        unsuccessful_iterate = true
        set_alpha!(solver, _alpha(solver)/2)
        _cnt(solver).l += 1
        if _alpha(solver) < alpha_min
            @debug(_logger(solver),
                    "Cannot find an acceptable step at iteration $(_cnt(solver).k). Switching to restoration phase.")
            _cnt(solver).k+=1
            return RESTORE
        else
            @trace(_logger(solver),"Step rejected; proceed with the next trial step.")
            if _alpha(solver) * norm(primal(_d(solver))) < eps(T)*10
                if (_cnt(solver).restoration_fail_count += 1) >= 4
                    return _cnt(solver).acceptable_cnt >0 ?
                        SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
                else
                    # (experimental) while giving up directly
                    # we give MadNLP.jl second chance to explore
                    # some possibility at the current iterate

                    fill!(_y(solver), zero(T))
                    fill!(_zl_r(solver), one(T))
                    fill!(_zu_r(solver), one(T))
                    empty!(_filter(solver))
                    push!(_filter(solver),(_theta_max(solver),-Inf))
                    _cnt(solver).k+=1

                    return REGULAR
                end
            end
        end
    end

    # this implements the heuristics in Section 3.2 of Ipopt paper.
    # Case I is only implemented
    if unsuccessful_iterate
        if (_cnt(solver).unsuccessful_iterate += 1) >= 4
            if _theta_max(solver)/10 > theta_trial
                @debug(_logger(solver), "restarting filter")
                set_theta_max!(solver, _theta_max(solver)/10)
                empty!(_filter(solver))
                push!(_filter(solver),(_theta_max(solver),-Inf))
            end
            _cnt(solver).unsuccessful_iterate = 0
        end
    else
        _cnt(solver).unsuccessful_iterate = 0
    end

    if !switching_condition || !armijo_condition
        @trace(_logger(solver),"Augmenting filter.")
        augment_filter!(_filter(solver),theta_trial,varphi_trial,_opt(solver).gamma_theta)
    end

    return LINESEARCH_SUCCEEDED
end

#=
    Variant of filter line-search for feasibility restoration phase.
=#

function filter_line_search_RR!(solver::AbstractMadNLPSolver{T}) where T
    RR = _RR(solver)
    theta_R = get_theta_R(_c(solver),RR.pp,RR.nn)
    varphi_R = get_varphi_R(RR.obj_val_R,_x_lr(solver),_xl_r(solver),_xu_r(solver),_x_ur(solver),RR.pp,RR.nn,RR.mu_R)
    varphi_d_R = get_varphi_d_R(
        RR.f_R,
        primal(_x(solver)),
        primal(_xl(solver)),
        primal(_xu(solver)),
        primal(_d(solver)),
        RR.pp,RR.nn,RR.dpp,RR.dnn,RR.mu_R,_opt(solver).rho,
    )

    # set alpha_min
    alpha_max = get_alpha_max_R(
        primal(_x(solver)),
        primal(_xl(solver)),
        primal(_xu(solver)),
        primal(_d(solver)),
        RR.pp,RR.dpp,RR.nn,RR.dnn,RR.tau_R,
    )
    set_alpha_z!(solver, get_alpha_z_R(_zl_r(solver),_zu_r(solver),dual_lb(_d(solver)),dual_ub(_d(solver)),RR.zp,RR.dzp,RR.zn,RR.dzn,RR.tau_R))
    alpha_min = get_alpha_min(theta_R,varphi_d_R,_theta_min(solver),_opt(solver).gamma_theta,_opt(solver).gamma_phi,
                                _opt(solver).alpha_min_frac,_opt(solver).delta,_opt(solver).s_theta,_opt(solver).s_phi)

    set_alpha!(solver, alpha_max)
    _cnt(solver).l = 1
    theta_R_trial = zero(T)
    varphi_R_trial = zero(T)
    small_search_norm = get_rel_search_norm(primal(_x(solver)), primal(_d(solver))) < 10*eps(T)
    switching_condition = is_switching(varphi_d_R,_alpha(solver),_opt(solver).s_phi,_opt(solver).delta,theta_R,_opt(solver).s_theta)
    armijo_condition = false

    while true
        copyto!(full(_x_trial(solver)), full(_x(solver)))
        copyto!(RR.pp_trial,RR.pp)
        copyto!(RR.nn_trial,RR.nn)
        axpy!(_alpha(solver),primal(_d(solver)),primal(_x_trial(solver)))
        axpy!(_alpha(solver),RR.dpp,RR.pp_trial)
        axpy!(_alpha(solver),RR.dnn,RR.nn_trial)

        RR.obj_val_R_trial = get_obj_val_R(
            RR.pp_trial,RR.nn_trial,RR.D_R,primal(_x_trial(solver)),RR.x_ref,_opt(solver).rho,RR.zeta)
        eval_cons_wrapper!(solver, _c_trial(solver), _x_trial(solver))
        theta_R_trial  = get_theta_R(_c_trial(solver),RR.pp_trial,RR.nn_trial)
        varphi_R_trial = get_varphi_R(
            RR.obj_val_R_trial,_x_trial_lr(solver),_xl_r(solver),_xu_r(solver),_x_trial_ur(solver),RR.pp_trial,RR.nn_trial,RR.mu_R)

        armijo_condition = is_armijo(varphi_R_trial,varphi_R,_opt(solver).eta_phi,_alpha(solver),varphi_d_R)

        small_search_norm && break
        set_ftype!(solver, get_ftype(
            RR.filter,theta_R,theta_R_trial,varphi_R,varphi_R_trial,
            switching_condition,armijo_condition,
            _theta_min(solver),_opt(solver).obj_max_inc,_opt(solver).gamma_theta,_opt(solver).gamma_phi,
            has_constraints(solver))
                )
        _ftype(solver) in ["f","h"] && (@trace(_logger(solver),"Step accepted with type $(_ftype(solver))"); break)

        set_alpha!(solver, _alpha(solver)/2)
        _cnt(solver).l += 1
        if _alpha(solver) < alpha_min
            @debug(_logger(solver),"Restoration phase cannot find an acceptable step at iteration $(_cnt(solver).k).")
            if (_cnt(solver).restoration_fail_count += 1) >= 4
                return RESTORATION_FAILED
            else
                # (experimental) while giving up directly
                # we give MadNLP.jl second chance to explore
                # some possibility at the current iterate

                fill!(_y(solver), zero(T))
                fill!(_zl_r(solver), one(T))
                fill!(_zu_r(solver), one(T))
                empty!(_filter(solver))
                push!(_filter(solver),(_theta_max(solver),-Inf))

                _cnt(solver).k+=1
                _cnt(solver).t+=1
                return REGULAR
            end
        else
            @trace(_logger(solver),"Step rejected; proceed with the next trial step.")
            _alpha(solver) < eps(T)*10 && return _cnt(solver).acceptable_cnt >0 ?
                SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
        end
    end

    if !switching_condition || !armijo_condition
        @trace(_logger(solver),"Augmenting restoration phase filter.")
        augment_filter!(RR.filter,theta_R_trial,varphi_R_trial,_opt(solver).gamma_theta)
    end

    return LINESEARCH_SUCCEEDED
end
