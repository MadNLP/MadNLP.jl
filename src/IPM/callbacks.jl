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

function eval_lag_hess_wrapper!(
    solver::MadNLPSolver,
    kkt::AbstractDenseKKTSystem{T, VT, MT, QN},
    x::Vector{T},
    l::Vector{T};
    is_resto=false,
) where {T, VT, MT, QN<:ExactHessian}
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
    kkt::AbstractKKTSystem{T, VT, MT, QN},
    x::Vector{T},
    l::Vector{T};
    is_resto=false,
) where {T, VT, MT<:AbstractMatrix{T}, QN<:AbstractQuasiNewton{T, VT}}
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger, "Update BFGS matrices.")

    qn = kkt.qn
    Bk = kkt.hess
    sk, yk = qn.sk, qn.yk
    n = length(qn.sk)
    m = size(kkt.jac, 1)

    # Code for GPU support
    if VT != Vector{T}
        # Load the buffers to transfer data between the host and the device.
        l_g = get!(kkt.etc, :l_g, VT(undef, m))
        x_g = get!(kkt.etc, :x_g, zeros(n))
        j_g = get!(kkt.etc, :j_g, zeros(n))
        # Init buffers.
        copyto!(l_g, l)
        copyto!(x_g, qn.last_x)
        fill!(j_g, zero(T))
    else
        l_g = l
        x_g = qn.last_x
        j_g = qn.last_jv
    end

    if cnt.obj_grad_cnt >= 2
        # Build sk = x+ - x
        copyto!(sk, 1, solver.x, 1, n)   # sₖ = x₊
        axpy!(-one(T), qn.last_x, sk)    # sₖ = x₊ - x

        # Build yk = ∇L+ - ∇L
        copyto!(yk, 1, solver.f, 1, n)   # yₖ = ∇f₊
        axpy!(-one(T), qn.last_g, yk)    # yₖ = ∇f₊ - ∇f
        if m > 0
            jtprod!(solver.jacl, kkt, l_g)
            yk .+= @view solver.jacl[1:n]         # yₖ += J₊ᵀ l₊
            NLPModels.jtprod!(nlp, x_g, l, j_g)
            copyto!(qn.last_jv, j_g)
            axpy!(-one(T), qn.last_jv, yk)        # yₖ += J₊ᵀ l₊ - Jᵀ l₊
        end

        if cnt.obj_grad_cnt == 2
            init!(qn, Bk, sk, yk)
        end
        update!(qn, Bk, sk, yk)
    end

    # Backup data for next step
    copyto!(qn.last_x, 1, solver.x, 1, n)
    copyto!(qn.last_g, 1, solver.f, 1, n)

    compress_hessian!(kkt)
    return get_hessian(kkt)
end

