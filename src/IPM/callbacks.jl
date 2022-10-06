function eval_f_wrapper(solver::MadNLPSolver, x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating objective.")
    x_ = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    cnt.eval_function_time += @elapsed begin
        sense = (get_minimize(nlp) ? one(T) : -one(T))
        obj_val = sense * obj(nlp, x_)
    end
    cnt.obj_cnt += 1
    if cnt.obj_cnt == 1 && !is_valid(obj_val)
        throw(InvalidNumberException(:obj))
    end
    return obj_val * solver.obj_scale[]
end

function eval_grad_f_wrapper!(solver::MadNLPSolver, f::Vector{T}, x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating objective gradient.")
    scale = solver.obj_scale[] * (get_minimize(nlp) ? one(T) : -one(T))
    x_ = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    f_ = _madnlp_unsafe_wrap(f, get_nvar(nlp))
    cnt.eval_function_time += @elapsed grad!(nlp, x_, f_)
    f .*= scale
    cnt.obj_grad_cnt += 1
    if cnt.obj_grad_cnt == 1 && !is_valid(f)
        throw(InvalidNumberException(:grad))
    end
    return f
end

function eval_cons_wrapper!(solver::MadNLPSolver, c::Vector{T}, x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger, "Evaluating constraints.")
    x_ = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    c_ = _madnlp_unsafe_wrap(c, get_ncon(nlp))
    cnt.eval_function_time += @elapsed cons!(nlp, x_, c_)
    view(c,solver.ind_ineq) .-= view(x,get_nvar(nlp)+1:solver.n)
    c .-= solver.rhs
    c .*= solver.con_scale
    cnt.con_cnt += 1
    if cnt.con_cnt == 1 && !is_valid(c)
        throw(InvalidNumberException(:cons))
    end
    return c
end

function eval_jac_wrapper!(solver::MadNLPSolver, kkt::AbstractKKTSystem, x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    ns = length(solver.ind_ineq)
    @trace(solver.logger, "Evaluating constraint Jacobian.")
    x_ = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed jac_coord!(
        nlp,
        x_,
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

function eval_lag_hess_wrapper!(solver::MadNLPSolver, kkt::AbstractKKTSystem, x::Vector{T},l::Vector{T};is_resto=false) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating Lagrangian Hessian.")
    dual(solver._w1) .= l .* solver.con_scale
    x_ = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    hess = get_hessian(kkt)
    scale = (get_minimize(nlp) ? one(T) : -one(T))
    scale *= (is_resto ? zero(T) : solver.obj_scale[])
    cnt.eval_function_time += @elapsed hess_coord!(
        nlp,
        x_,
        dual(solver._w1),
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

function eval_jac_wrapper!(solver::MadNLPSolver, kkt::AbstractDenseKKTSystem, x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    ns = length(solver.ind_ineq)
    @trace(solver.logger, "Evaluating constraint Jacobian.")
    x_ = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed jac_dense!(
        nlp,
        x_,
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

function eval_lag_hess_wrapper!(solver::MadNLPSolver, kkt::AbstractDenseKKTSystem, x::Vector{T},l::Vector{T};is_resto=false) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating Lagrangian Hessian.")
    dual(solver._w1) .= l .* solver.con_scale
    x_ = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    hess = get_hessian(kkt)
    scale = (get_minimize(nlp) ? one(T) : -one(T))
    scale *= (is_resto ? zero(T) : solver.obj_scale[])
    cnt.eval_function_time += @elapsed hess_dense!(
        nlp,
        x_,
        dual(solver._w1),
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

