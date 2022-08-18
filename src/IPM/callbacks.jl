function eval_f_wrapper(solver::MadNLPSolver, x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating objective.")
    x_nlpmodel = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    cnt.eval_function_time += @elapsed obj_val = (get_minimize(nlp) ? 1. : -1.) * obj(nlp,x_nlpmodel)
    cnt.obj_cnt+=1
    cnt.obj_cnt==1 && (is_valid(obj_val) || throw(InvalidNumberException()))
    return obj_val*solver.obj_scale[]
end

function eval_grad_f_wrapper!(solver::MadNLPSolver, f::Vector{T},x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating objective gradient.")
    x_nlpmodel = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    f_nlpmodel = _madnlp_unsafe_wrap(f, get_nvar(nlp))
    cnt.eval_function_time += @elapsed grad!(
        nlp,
        x_nlpmodel,
        f_nlpmodel
    )
    f.*=solver.obj_scale[] * (get_minimize(nlp) ? 1. : -1.)
    cnt.obj_grad_cnt+=1
    cnt.obj_grad_cnt==1 && (is_valid(f)  || throw(InvalidNumberException()))
    return f
end

function eval_cons_wrapper!(solver::MadNLPSolver, c::Vector{T},x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger, "Evaluating constraints.")
    x_nlpmodel = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    c_nlpmodel = _madnlp_unsafe_wrap(c, get_ncon(nlp))
    cnt.eval_function_time += @elapsed cons!(
        nlp,
        x_nlpmodel,
        c_nlpmodel
    )
    view(c,solver.ind_ineq).-=view(x,get_nvar(nlp)+1:solver.n)
    c.-=solver.rhs
    c.*=solver.con_scale
    cnt.con_cnt+=1
    cnt.con_cnt==2 && (is_valid(c) || throw(InvalidNumberException()))
    return c
end

function eval_jac_wrapper!(solver::MadNLPSolver, kkt::AbstractKKTSystem, x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    ns = length(solver.ind_ineq)
    @trace(solver.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    x_nlpmodel = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    jac_nlpmodel = _madnlp_unsafe_wrap(jac, get_nnzj(nlp.meta))
    cnt.eval_function_time += @elapsed jac_coord!(
        nlp,
        x_nlpmodel, 
        jac_nlpmodel
    )
    compress_jacobian!(kkt)
    cnt.con_jac_cnt+=1
    cnt.con_jac_cnt==1 && (is_valid(jac) || throw(InvalidNumberException()))
    @trace(solver.logger,"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(solver::MadNLPSolver, kkt::AbstractKKTSystem, x::Vector{T},l::Vector{T};is_resto=false) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating Lagrangian Hessian.")
    dual(solver._w1) .= l.*solver.con_scale
    hess = get_hessian(kkt)
    x_nlpmodel = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    hess_nlpmodel = _madnlp_unsafe_wrap(hess, get_nnzh(nlp.meta))
    cnt.eval_function_time += @elapsed hess_coord!(
        nlp,
        x_nlpmodel,
        dual(solver._w1),
        hess_nlpmodel;
        obj_weight = (get_minimize(nlp) ? 1. : -1.) * (is_resto ? 0.0 : solver.obj_scale[])
    )
    compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1
    cnt.lag_hess_cnt==1 && (is_valid(hess) || throw(InvalidNumberException()))
    return hess
end

function eval_jac_wrapper!(solver::MadNLPSolver, kkt::AbstractDenseKKTSystem, x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    ns = length(solver.ind_ineq)
    @trace(solver.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    x_nlpmodel = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    cnt.eval_function_time += @elapsed jac_dense!(
        nlp,
        x_nlpmodel,
        jac
    )
    compress_jacobian!(kkt)
    cnt.con_jac_cnt+=1
    cnt.con_jac_cnt==1 && (is_valid(jac) || throw(InvalidNumberException()))
    @trace(solver.logger,"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(solver::MadNLPSolver, kkt::AbstractDenseKKTSystem, x::Vector{T},l::Vector{T};is_resto=false) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating Lagrangian Hessian.")
    dual(solver._w1) .= l.*solver.con_scale
    hess = get_hessian(kkt)
    x_nlpmodel = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    cnt.eval_function_time += @elapsed hess_dense!(
        nlp,
        x_nlpmodel,
        dual(solver._w1),
        hess;
        obj_weight = (get_minimize(nlp) ? 1. : -1.) * (is_resto ? 0.0 : solver.obj_scale[])
    )
    compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1
    cnt.lag_hess_cnt==1 && (is_valid(hess) || throw(InvalidNumberException()))
    return hess
end
