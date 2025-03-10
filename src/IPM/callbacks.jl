function eval_f_wrapper(solver::AbstractMadNLPSolver, x::PrimalVector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating objective.")
    cnt.eval_function_time += @elapsed begin
        sense = (get_minimize(nlp) ? one(T) : -one(T))
        obj_val = sense * _eval_f_wrapper(solver.cb, variable(x))
    end
    cnt.obj_cnt += 1
    if cnt.obj_cnt == 1 && !is_valid(obj_val)
        throw(InvalidNumberException(:obj))
    end
    return obj_val
end

function eval_grad_f_wrapper!(solver::AbstractMadNLPSolver, f::PrimalVector{T}, x::PrimalVector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating objective gradient.")
    cnt.eval_function_time += @elapsed _eval_grad_f_wrapper!(
        solver.cb,
        variable(x),
        variable(f),
    )
    if !get_minimize(nlp)
        variable(f) .*= -one(T)
    end
    cnt.obj_grad_cnt+=1

    if cnt.obj_grad_cnt == 1 && !is_valid(full(f))
        throw(InvalidNumberException(:grad))
    end
    return f
end

function eval_cons_wrapper!(solver::AbstractMadNLPSolver, c::AbstractVector{T}, x::PrimalVector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger, "Evaluating constraints.")
    cnt.eval_function_time += @elapsed _eval_cons_wrapper!(
        solver.cb,
        variable(x),
        c,
    )
    if solver.opt.slack_reset
        slack(x) .= view(c, solver.ind_ineq)
        view(c,solver.ind_ineq) .= 0.0
    else
        view(c,solver.ind_ineq) .-= slack(x)
    end

    c .-= solver.rhs
    cnt.con_cnt+=1
    if cnt.con_cnt == 1 && !is_valid(c)
        throw(InvalidNumberException(:cons))
    end
    return c
end

function eval_jac_wrapper!(solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, x::PrimalVector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    ns = length(solver.ind_ineq)
    @trace(solver.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed _eval_jac_wrapper!(
        solver.cb,
        variable(x),
        jac,
        )
    compress_jacobian!(kkt)
    cnt.con_jac_cnt += 1
    if cnt.con_jac_cnt == 1 && !is_valid(jac)
        throw(InvalidNumberException(:jac))
    end
    @trace(solver.logger,"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, x::PrimalVector{T},l::AbstractVector{T};is_resto=false) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating Lagrangian Hessian.")
    hess = get_hessian(kkt)
    scale = (get_minimize(nlp) ? one(T) : -one(T)) * (is_resto ? zero(T) : one(T))
    cnt.eval_function_time += @elapsed _eval_lag_hess_wrapper!(
        solver.cb,
        variable(x),
        l,
        hess;
        obj_weight = scale,
    )
    compress_hessian!(kkt)
    cnt.lag_hess_cnt += 1
    if cnt.lag_hess_cnt == 1 && !is_valid(hess)
        throw(InvalidNumberException(:hess))
    end
    return hess
end

function eval_jac_wrapper!(solver::AbstractMadNLPSolver, kkt::AbstractDenseKKTSystem, x::PrimalVector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    ns = length(solver.ind_ineq)
    @trace(solver.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed _eval_jac_wrapper!(
        solver.cb,
        variable(x),
        jac,
    )
    compress_jacobian!(kkt)
    cnt.con_jac_cnt+=1
    if cnt.con_jac_cnt == 1 && !is_valid(jac)
        throw(InvalidNumberException(:jac))
    end
    @trace(solver.logger,"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(
    solver::AbstractMadNLPSolver,
    kkt::AbstractDenseKKTSystem{T, VT, MT, QN},
    x::PrimalVector{T},
    l::AbstractVector{T};
    is_resto=false,
) where {T, VT, MT, QN<:ExactHessian}
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating Lagrangian Hessian.")
    hess = get_hessian(kkt)
    scale = is_resto ? zero(T) : get_minimize(nlp) ? one(T) : -one(T)
    cnt.eval_function_time += @elapsed _eval_lag_hess_wrapper!(
        solver.cb,
        variable(x),
        l,
        hess;
        obj_weight = scale,
    )
    compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1
    if cnt.lag_hess_cnt == 1 && !is_valid(hess)
        throw(InvalidNumberException(:hess))
    end
    return hess
end

function eval_lag_hess_wrapper!(
    solver::AbstractMadNLPSolver,
    kkt::AbstractKKTSystem{T, VT, MT, QN},
    x::PrimalVector{T},
    l::AbstractVector{T};
    is_resto=false,
) where {T, VT, MT<:AbstractMatrix{T}, QN<:AbstractQuasiNewton{T, VT}}
    cb = solver.cb
    cnt = solver.cnt
    @trace(solver.logger, "Update BFGS matrices.")

    qn = kkt.quasi_newton
    Bk = kkt.hess
    sk, yk = qn.sk, qn.yk
    n = length(qn.sk)
    m = size(kkt.jac, 1)

    if cnt.obj_grad_cnt >= 2
        # Build sk = x+ - x
        copyto!(sk, 1, variable(solver.x), 1, n)   # sₖ = x₊
        axpy!(-one(T), qn.last_x, sk)              # sₖ = x₊ - x
        # Build yk = ∇L+ - ∇L
        copyto!(yk, 1, variable(solver.f), 1, n)   # yₖ = ∇f₊
        axpy!(-one(T), qn.last_g, yk)              # yₖ = ∇f₊ - ∇f
        if m > 0
            jtprod!(solver.jacl, kkt, l)
            yk .+= @view(solver.jacl[1:n])         # yₖ += J₊ᵀ l₊
            _eval_jtprod_wrapper!(cb, qn.last_x, l, qn.last_jv)
            axpy!(-one(T), qn.last_jv, yk)         # yₖ += J₊ᵀ l₊ - Jᵀ l₊
        end
        # Update quasi-Newton approximation.
        update!(qn, Bk, sk, yk)
    else
        # Init quasi-Newton approximation
        g0 = variable(solver.f)
        f0 = solver.obj_val
        init!(qn, Bk, g0, f0)
    end

    # Backup data for next step
    copyto!(qn.last_x, 1, variable(solver.x), 1, n)
    copyto!(qn.last_g, 1, variable(solver.f), 1, n)

    compress_hessian!(kkt)
    return get_hessian(kkt)
end

