function eval_f_wrapper(solver::MadNLPSolver, x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating objective.")
    x_nlpmodel = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    cnt.eval_function_time += @elapsed obj_val = (get_minimize(nlp) ? 1. : -1.) * obj(nlp,x_nlpmodel)
    cnt.obj_cnt+=1
    cnt.obj_cnt==1 && (is_valid(obj_val) || throw(InvalidNumberException(:obj)))
    return obj_val*solver.obj_scale[]
end

function eval_grad_f_wrapper!(solver::MadNLPSolver, f::Vector{T},x::Vector{T}) where T
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger,"Evaluating objective gradient.")
    obj_scaling = solver.obj_scale[] * (get_minimize(nlp) ? one(T) : -one(T))
    x_nlpmodel = _madnlp_unsafe_wrap(x, get_nvar(nlp))
    f_nlpmodel = _madnlp_unsafe_wrap(f, get_nvar(nlp))
    cnt.eval_function_time += @elapsed grad!(
        nlp,
        x_nlpmodel,
        f_nlpmodel
    )
    _scal!(obj_scaling, f)
    cnt.obj_grad_cnt+=1
    cnt.obj_grad_cnt==1 && (is_valid(f)  || throw(InvalidNumberException(:grad)))
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
    c .-= solver.rhs
    c .*= solver.con_scale
    cnt.con_cnt+=1
    cnt.con_cnt==2 && (is_valid(c) || throw(InvalidNumberException(:cons)))
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
    cnt.con_jac_cnt==1 && (is_valid(jac) || throw(InvalidNumberException(:jac)))
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
    cnt.lag_hess_cnt==1 && (is_valid(hess) || throw(InvalidNumberException(:hess)))
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
    cnt.con_jac_cnt==1 && (is_valid(jac) || throw(InvalidNumberException(:jac)))
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
    cnt.lag_hess_cnt==1 && (is_valid(hess) || throw(InvalidNumberException(:hess)))
    return hess
end

function eval_lag_hess_wrapper!(
    solver::MadNLPSolver,
    kkt::BFGSKKTSystem,
    x::Vector{Float64},
    l::Vector{Float64};
    is_resto=false,
)
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger, "Update BFGS matrices.")
    Bk = kkt.hess
    n, p = size(Bk)

    # Make sure we have evaluated the gradient before.
    if cnt.obj_grad_cnt > 1
        sk = kkt.sk
        # Build sk = x+ - x
        copyto!(sk, 1, solver.x, 1, n)
        axpy!(-1.0, kkt.last_x, sk)

        # Build yk = ∇L+ - ∇L
        yk = kkt.yk
        copyto!(yk, 1, solver.f, 1, n)
        mul!(yk, kkt.jac', l, 1.0, 1.0)
        axpy!(-1.0, kkt.last_grad, yk)
        mul!(yk, kkt.jac_prev', l, -1.0, 1.0)

        # Initial update (Nocedal & Wright, p.143)
        if cnt.obj_grad_cnt == 2
            yksk = dot(yk, sk)
            sksk = dot(sk, sk)
            Bk[diagind(Bk)] .= yksk ./ sksk
        end

        update!(kkt.Bk, Bk, sk, yk)
    end

    copyto!(kkt.last_x, 1, solver.x, 1, n)
    copyto!(kkt.last_grad, 1, solver.f, 1, n)
    copyto!(kkt.jac_prev, solver.kkt.jac)

    compress_hessian!(kkt)
    return get_hessian(kkt)
end

