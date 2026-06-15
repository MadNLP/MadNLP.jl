# IPM-scoped timing helpers. AbstractOptions, MadNLPLogger, BLAS macros,
# MadNLPCounters, _madnlp_unsafe_wrap, SubVector etc. live in MadCore.

"""
    timing_callbacks(ips::InteriorPointSolver; ntrials=10)

Return the average timings spent in each callback for `ntrials` different trials.
Results are returned inside a named-tuple.
"""
function timing_callbacks(ips; ntrials = 10)
    t_f, t_c, t_g, t_j, t_h = (0.0, 0.0, 0.0, 0.0, 0.0)
    for _ in 1:ntrials
        t_f += @elapsed eval_f_wrapper(ips, ips.x)
        t_c += @elapsed eval_cons_wrapper!(ips, ips.c, ips.x)
        t_g += @elapsed eval_grad_f_wrapper!(ips, ips.f, ips.x)
        t_j += @elapsed eval_jac_wrapper!(ips, ips.kkt, ips.x)
        t_h += @elapsed eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.y)
    end
    return (
        time_eval_objective = t_f / ntrials,
        time_eval_constraints = t_c / ntrials,
        time_eval_gradient = t_g / ntrials,
        time_eval_jacobian = t_j / ntrials,
        time_eval_hessian = t_h / ntrials,
    )
end

"""
    timing_linear_solver(ips::InteriorPointSolver; ntrials=10)

Return the average timings spent in the linear solver for `ntrials` different trials.
Results are returned inside a named-tuple.
"""
function timing_linear_solver(ips; ntrials = 10)
    t_build, t_factorize, t_backsolve = (0.0, 0.0, 0.0)
    for _ in 1:ntrials
        t_build += @elapsed build_kkt!(ips.kkt)
        t_factorize += @elapsed factorize_kkt!(ips.kkt)
        t_backsolve += @elapsed solve_kkt!(ips.kkt, ips.d)
    end
    return (
        time_build_kkt = t_build / ntrials,
        time_factorization = t_factorize / ntrials,
        time_backsolve = t_backsolve / ntrials,
    )
end

"""
    timing_madnlp(ips::InteriorPointSolver; ntrials=10)

Return the average time spent in the callbacks and in the linear solver,
for `ntrials` different trials. Results are returned as a named-tuple.
"""
function timing_madnlp(ips; ntrials = 10)
    return (
        time_linear_solver = timing_linear_solver(ips; ntrials = ntrials),
        time_callbacks = timing_callbacks(ips; ntrials = ntrials),
    )
end
