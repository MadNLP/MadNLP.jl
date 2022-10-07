function eval_f_wrapper(solver::MadNLPSolver, x::PrimalVector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating objective.")
    cnt.eval_function_time += @elapsed begin
        sense = (get_minimize(nlp) ? one(T) : -one(T))
        obj_val = sense * obj(nlp, variable(x))
    end
    cnt.obj_cnt += 1
    if cnt.obj_cnt == 1 && !is_valid(obj_val)
        throw(InvalidNumberException(:obj))
    end
    return obj_val * solver.obj_scale[]
end

function eval_grad_f_wrapper!(solver::MadNLPSolver, f::PrimalVector{T}, x::PrimalVector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating objective gradient.")
    scale = solver.obj_scale[] * (get_minimize(nlp) ? one(T) : -one(T))
    cnt.eval_function_time += @elapsed grad!(nlp, variable(x), variable(f))
    grad!(nlp, variable(x), variable(f))
    full(f) .*= scale
    cnt.obj_grad_cnt += 1
    if cnt.obj_grad_cnt == 1 && !is_valid(full(f))
        throw(InvalidNumberException(:grad))
    end
    return f
end

function eval_cons_wrapper!(solver::MadNLPSolver, c::Vector{T}, x::PrimalVector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger, "Evaluating constraints.")
    cnt.eval_function_time += @elapsed cons!(nlp, variable(x), c)
    view(c,solver.ind_ineq) .-= slack(x)
    c .-= solver.rhs
    c .*= solver.con_scale
    cnt.con_cnt += 1
    if cnt.con_cnt == 1 && !is_valid(c)
        throw(InvalidNumberException(:cons))
    end
    return c
end

function eval_jac_wrapper!(solver::MadNLPSolver, kkt::AbstractKKTSystem, x::PrimalVector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    ns = length(solver.ind_ineq)
    @trace(solver.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed jac_coord!(
        nlp,
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

function eval_lag_hess_wrapper!(solver::MadNLPSolver, kkt::AbstractKKTSystem, x::PrimalVector{T},l::Vector{T};is_resto=false) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating Lagrangian Hessian.")
    dual(solver._w1) .= l .* solver.con_scale
    hess = get_hessian(kkt)
    scale = (get_minimize(nlp) ? one(T) : -one(T))
    scale *= (is_resto ? zero(T) : solver.obj_scale[])
    cnt.eval_function_time += @elapsed hess_coord!(
        nlp,
        variable(x),
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

function eval_jac_wrapper!(solver::MadNLPSolver, kkt::AbstractDenseKKTSystem, x::PrimalVector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    ns = length(solver.ind_ineq)
    @trace(solver.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed jac_dense!(
        nlp,
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

function eval_lag_hess_wrapper!(solver::MadNLPSolver, kkt::AbstractDenseKKTSystem, x::PrimalVector{T},l::Vector{T};is_resto=false) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating Lagrangian Hessian.")
    dual(solver._w1) .= l .* solver.con_scale
    hess = get_hessian(kkt)
    scale = (get_minimize(nlp) ? one(T) : -one(T))
    scale *= (is_resto ? zero(T) : solver.obj_scale[])
    cnt.eval_function_time += @elapsed hess_dense!(
        nlp,
        variable(x),
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

