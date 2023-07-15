abstract type AbstractCallback{T,VT} end

abstract type AbstractFixedVariableHandler end
struct MakeParameter{VT,VI} <: AbstractFixedVariableHandler
    fixed::VI
    fixedj::VI
    fixedh::VI
    grad_storage::VT
end
struct RelaxBound <: AbstractFixedVariableHandler end

struct SparseCallback{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    I <: AbstractNLPModel{T, VT},
    FH <: AbstractFixedVariableHandler,
    } <: AbstractCallback{T, VT}
    
    nlp::I
    nvar::Int
    ncon::Int
    nnzj::Int
    nnzh::Int

    con_buffer::VT
    jac_buffer::VT
    grad_buffer::VT
    hess_buffer::VT

    jac_I::VI
    jac_J::VI
    hess_I::VI
    hess_J::VI

    obj_scale::VT
    con_scale::VT
    jac_scale::VT

    fixed_handler::FH
end

struct DenseCallback{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    I <: AbstractNLPModel{T, VT},
    FH <: AbstractFixedVariableHandler,
    } <: AbstractCallback{T, VT}
    
    nlp::I
    nvar::Int
    ncon::Int

    con_buffer::VT
    jac_buffer::MT
    grad_buffer::VT

    obj_scale::VT
    con_scale::VT

    fixed_handler::FH
end


create_array(cb::AbstractCallback, args...) = similar(get_x0(cb.nlp), args...)

function set_obj_scale!(obj_scale, f::VT,max_gradient) where {T, VT <: AbstractVector{T}}
    obj_scale[] = min(one(T), max_gradient / norm(f,Inf))
end
function set_con_scale_sparse!(con_scale::VT, jac_I,jac_buffer, max_gradient) where {T, VT <: AbstractVector{T}}
    fill!(con_scale, one(T))
    @inbounds @simd for i in 1:length(jac_I)
        row = jac_I[i]
        con_scale[row] = max(con_scale[row], abs(jac_buffer[i]))
    end
    @inbounds @simd for i in eachindex(con_scale)
        con_scale[i] = min(one(T), max_gradient / con_scale[i])
    end
end
function set_jac_scale_sparse!(jac_scale::VT, con_scale, jac_I) where {T, VT <: AbstractVector{T}}
    fill!(jac_scale, one(T))
    @inbounds @simd for i in 1:length(jac_I)
        index = jac_I[i]
        jac_scale[i] = con_scale[index]
    end
end
function set_con_scale_dense!(con_scale::VT, jac_buffer, max_gradient) where {T, VT <: AbstractVector{T}}
    con_scale .= min.(one(T), max_gradient ./ mapreduce(abs, max, jac_buffer, dims=2, init=one(T)))
end


function create_dense_fixed_handler(
    fixed_variable_treatment::Type{MakeParameter},
    nlp,
    opt
    )
    lvar = get_lvar(nlp)
    uvar = get_uvar(nlp)
    
    isfixed  = (lvar .== uvar)
    fixed  = findall(isfixed)

    return MakeParameter(
        fixed,
        similar(fixed,0),
        similar(fixed,0),
        similar(lvar, length(fixed))
    )
end

function create_sparse_fixed_handler(
    fixed_variable_treatment::Type{MakeParameter},
    nlp,
    jac_I,
    jac_J,
    hess_I,
    hess_J;
    opt
    )
    lvar = get_lvar(nlp)
    uvar = get_uvar(nlp)
    nnzj = get_nnzj(nlp.meta)
    nnzh = get_nnzh(nlp.meta)
    
    isfixed  = (lvar .== uvar)
    
    fixed  = findall(isfixed)
    fixedj = findall(@view(isfixed[jac_J]))
    fixedh = findall(@view(isfixed[hess_I]) .|| @view(isfixed[hess_J]))
    nfixed = length(fixed)

    nnzh = nnzh + nfixed
    resize!(hess_I, nnzh)
    resize!(hess_J, nnzh)
    copyto!(@view(hess_I[end-nfixed+1:end]), fixed)
    copyto!(@view(hess_J[end-nfixed+1:end]), fixed)

    fixed_handler = MakeParameter(
        fixed,
        fixedj,
        fixedh,
        similar(lvar, length(fixed))
    )
    
    return fixed_handler, nnzj, nnzh
end

function create_sparse_fixed_handler(
    fixed_variable_treatment::Type{RelaxBound},
    nlp,
    jac_I,
    jac_J,
    hess_I,
    hess_J;
    opt
    )

    fixed_handler = RelaxBound()

    
    return fixed_handler, get_nnzj(nlp.meta), get_nnzh(nlp.meta)
end

function create_callback(
    ::Type{SparseCallback},
    nlp::AbstractNLPModel{T, VT},
    opt,
    ) where {T, VT}

    n = get_nvar(nlp)
    m = get_ncon(nlp)
    nnzj = get_nnzj(nlp.meta)
    nnzh = get_nnzh(nlp.meta)
    

    x0   = get_x0(nlp)

    con_buffer = similar(x0, m)
    grad_buffer = similar(x0, n)
    jac_buffer = similar(x0, nnzj)
    hess_buffer = similar(x0, nnzh)

    jac_I = similar(x0, Int, nnzj)
    jac_J = similar(x0, Int, nnzj)
    hess_I = similar(x0, Int, nnzh)
    hess_J = similar(x0, Int, nnzh)

    obj_scale = similar(jac_buffer, 1)
    con_scale = similar(jac_buffer, m)
    jac_scale = similar(jac_buffer, nnzj)
    
    NLPModels.jac_structure!(nlp,jac_I,jac_J)
    NLPModels.hess_structure!(nlp,hess_I,hess_J)

    fixed_handler, nnzj, nnzh = create_sparse_fixed_handler(
        opt.fixed_variable_treatment,
        nlp,
        jac_I, jac_J, hess_I, hess_J;
        opt = opt
    )

    return SparseCallback(
        nlp,
        n,m,nnzj,nnzh,
        con_buffer,
        jac_buffer,
        grad_buffer,
        hess_buffer,
        jac_I,
        jac_J,
        hess_I,
        hess_J,
        obj_scale,
        con_scale,
        jac_scale,
        fixed_handler
    )
end

function create_callback(
    ::Type{DenseCallback},
    nlp::AbstractNLPModel{T, VT},
    opt,
    ) where {T, VT}

    n = get_nvar(nlp)
    m = get_ncon(nlp)
    
    x0   = similar(get_x0(nlp))
    con_buffer = similar(x0, m)
    jac_buffer = similar(x0, m, n)
    grad_buffer = similar(x0, n)
    obj_scale = similar(x0, 1)
    con_scale = similar(x0, m)

    fixed_handler = create_dense_fixed_handler(
        opt.fixed_variable_treatment,
        nlp,
        opt
    )

    return DenseCallback(
        nlp,
        n,m,
        con_buffer,
        jac_buffer,
        grad_buffer,
        obj_scale,
        con_scale,
        fixed_handler
    )
end

function _treat_fixed_variable_initialize!(fixed_handler::RelaxBound, x0, lvar, uvar) end
function _treat_fixed_variable_initialize!(fixed_handler::MakeParameter, x0, lvar, uvar)
    fixed = fixed_handler.fixed
    copyto!(@view(x0[fixed]), @view(lvar[fixed]))
    fill!(@view(lvar[fixed]), -Inf)
    fill!(@view(uvar[fixed]),  Inf)
end
function initialize!(
    cb::AbstractCallback,
    x, xl, xu, y0, rhs,
    ind_ineq,
    opt
    )

    x0= variable(x)
    lvar= variable(xl)
    uvar= variable(xu)
    
    fixed_handler = cb.fixed_handler
    nlp = cb.nlp
    
    con_buffer =cb.con_buffer
    grad_buffer =cb.grad_buffer

    lcon = get_lcon(nlp)
    ucon = get_ucon(nlp)
    
    x0   .= get_x0(nlp)
    y0   .= get_y0(nlp) 
    lvar .= get_lvar(nlp)
    uvar .= get_uvar(nlp)
    lcon .= get_lcon(nlp) 
    ucon .= get_ucon(nlp)
    _treat_fixed_variable_initialize!(fixed_handler, x0, lvar, uvar)


    set_initial_bounds!(
        lvar,
        uvar,
        opt.tol
    )
    initialize_variables!(
        x0,
        lvar,
        uvar,
        opt.bound_push,
        opt.bound_fac
    )
    
    NLPModels.cons!(nlp,x0,con_buffer)
    is = findall(.!(lcon .== -Inf .&& ucon .== Inf) .&& (lcon .!= ucon))
    
    slack(xl) .= view(lcon, ind_ineq)
    slack(xu) .= view(ucon, ind_ineq)
    rhs .= (lcon.==ucon).*lcon
    copyto!(slack(x), @view(con_buffer[ind_ineq]))

    set_initial_bounds!(
        slack(xl),
        slack(xu),
        opt.tol
    )
    initialize_variables!(
        slack(x),
        slack(xl),
        slack(xu),
        opt.bound_push,
        opt.bound_fac
    )

end

function set_scaling!(
    cb::SparseCallback,
    x, xl, xu, y0, rhs,
    ind_ineq,
    nlp_scaling_max_gradient
    )
    
    x0= variable(x)

    nlp = cb.nlp
    lcon = get_lcon(nlp)
    ucon = get_ucon(nlp)
    obj_scale = cb.obj_scale
    con_scale = cb.con_scale
    jac_scale = cb.jac_scale    
    con_buffer =cb.con_buffer
    jac_buffer =cb.jac_buffer
    grad_buffer =cb.grad_buffer

    # Set scaling
    NLPModels.jac_coord!(nlp,x0,jac_buffer)
    set_con_scale_sparse!(con_scale, cb.jac_I, jac_buffer, nlp_scaling_max_gradient)
    set_jac_scale_sparse!(jac_scale, con_scale, cb.jac_I)
    
    NLPModels.grad!(nlp,x0,grad_buffer)
    set_obj_scale!(obj_scale, grad_buffer, nlp_scaling_max_gradient)

    con_scale_slk = @view(con_scale[ind_ineq])
    y0  ./= con_scale
    rhs .*= con_scale
    slack(x) .*= con_scale_slk
    slack(xl) .*= con_scale_slk
    slack(xu) .*= con_scale_slk
end

function set_scaling!(
    cb::DenseCallback,
    x, xl, xu, y0, rhs,
    ind_ineq,
    nlp_scaling_max_gradient
    )
    
    x0= variable(x)

    nlp = cb.nlp
    lcon = get_lcon(nlp)
    ucon = get_ucon(nlp)
    obj_scale = cb.obj_scale
    con_scale = cb.con_scale
    con_buffer =cb.con_buffer
    jac_buffer =cb.jac_buffer
    grad_buffer =cb.grad_buffer

    # Set scaling
    jac_dense!(nlp,x0,jac_buffer)
    set_con_scale_dense!(con_scale, jac_buffer, nlp_scaling_max_gradient)
    
    NLPModels.grad!(nlp,x0,grad_buffer)
    set_obj_scale!(obj_scale, grad_buffer, nlp_scaling_max_gradient)

    con_scale_slk = @view(con_scale[ind_ineq])
    y0  ./= con_scale
    rhs .*= con_scale
    slack(x) .*= con_scale_slk
    slack(xl) .*= con_scale_slk
    slack(xu) .*= con_scale_slk
end

function _jac_sparsity_wrapper!(
    cb::SparseCallback,
    I::AbstractVector,J::AbstractVector
    )
    
    copyto!(I, cb.jac_I)
    copyto!(J, cb.jac_J)
end

function _hess_sparsity_wrapper!(
    cb::SparseCallback,
    I::AbstractVector,J::AbstractVector
    ) 
    copyto!(I, cb.hess_I)
    copyto!(J, cb.hess_J)
end


function _eval_cons_wrapper!(cb::AbstractCallback,x::AbstractVector,c::AbstractVector)
    NLPModels.cons!(cb.nlp, x,c)
    c .*= cb.con_scale
    return c
end


function _eval_jac_wrapper!(
    cb::SparseCallback,
    x::AbstractVector,
    jac::AbstractVector
    )
    
    nnzj_orig = get_nnzj(cb.nlp.meta)
    NLPModels.jac_coord!(cb.nlp, x, jac)
    jac .*= cb.jac_scale

    _treat_fixed_variable_jac_coord!(cb.fixed_handler, cb, x, jac)
end
function _treat_fixed_variable_jac_coord!(fixed_handler::RelaxBound, cb, x, jac) end
function _treat_fixed_variable_jac_coord!(fixed_handler::MakeParameter, cb::SparseCallback{T}, x, jac) where T
    fill!(@view(jac[fixed_handler.fixedj]), zero(T))
end

function _eval_grad_f_wrapper!(
    cb::AbstractCallback{T},
    x::AbstractVector,
    grad::AbstractVector
    ) where T
    
    NLPModels.grad!(cb.nlp, x, grad)
    grad .*= cb.obj_scale[]
    _treat_fixed_variable_grad!(cb.fixed_handler, cb, x, grad)
end
function _treat_fixed_variable_grad!(fixed_handler::RelaxBound, cb, x, grad) end
function _treat_fixed_variable_grad!(fixed_handler::MakeParameter, cb::AbstractCallback{T}, x, grad) where T
    fixed_handler.grad_storage .= @view(grad[fixed_handler.fixed])
    map!(
        (x,y)->x-y,
        @view(grad[fixed_handler.fixed]),
        @view(x[cb.fixed_handler.fixed]),
        @view(get_lvar(cb.nlp)[cb.fixed_handler.fixed])
    )
end

function _eval_f_wrapper(cb::AbstractCallback,x::AbstractVector)
    return NLPModels.obj(cb.nlp,x)* cb.obj_scale[]
end

function _eval_lag_hess_wrapper!(
    cb::SparseCallback{T},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector; 
    obj_weight = 1.0
    ) where T
    
    nnzh_orig = get_nnzh(cb.nlp.meta)
    
    cb.con_buffer .= y .* cb.con_scale
    NLPModels.hess_coord!(
        cb.nlp, x, cb.con_buffer, view(hess, 1:nnzh_orig);
        obj_weight=obj_weight * cb.obj_scale[]
    )
    _treat_fixed_variable_hess_coord!(cb.fixed_handler, cb, hess) 
end

function _treat_fixed_variable_hess_coord!(fixed_handler::RelaxBound, cb, hess) end
function _treat_fixed_variable_hess_coord!(fixed_handler::MakeParameter, cb::SparseCallback{T}, hess) where T
    nnzh_orig = get_nnzh(cb.nlp.meta)
    fill!(@view(hess[fixed_handler.fixedh]), zero(T))
    fill!(@view(hess[nnzh_orig+1:end]), one(T))
end

function _eval_jac_wrapper!(
    cb::SparseCallback{T},
    x::AbstractVector,
    jac::AbstractMatrix
    ) where T
    
    jac_buffer = cb.jac_buffer
    _eval_jac_wrapper!(cb, x, jac_buffer)
    fill!(jac, zero(T))
    @inbounds @simd for k=1:length(cb.jac_I)
        i,j = cb.jac_I[k], cb.jac_J[k]
        jac[i,j] += jac_buffer[k]
    end
end

function _eval_lag_hess_wrapper!(
    cb::SparseCallback{T},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractMatrix;
    obj_weight = one(T)
    ) where T
    
    hess_buffer = cb.hess_buffer
    _eval_lag_hess_wrapper!(cb, x, y, hess_buffer; obj_weight=obj_weight * cb.obj_scale[])
    fill!(hess, zero(T))
    @inbounds @simd for k=1:length(cb.hess_I)
        i,j = cb.hess_I[k], cb.hess_J[k]
        hess[i,j] += hess_buffer[k]
    end
    _treat_fixed_variable_hess_dense!(cb.fixed_handler, cb, hess)
end
function _treat_fixed_variable_hess_dense!(fixed_handler::RelaxBound, cb, hess) end
function _treat_fixed_variable_hess_dense!(fixed_handler::MakeParameter, cb::SparseCallback{T}, hess) where T
    nnzh_orig = get_nnzh(cb.nlp.meta)
    
    for i in fixed_handler.fixed
        hess[i,i] = one(T)
    end
end


function _eval_jac_wrapper!(
    cb::DenseCallback{T},
    x::AbstractVector,
    jac::AbstractMatrix
    ) where T
    
    jac_dense!(cb.nlp, x, jac)
    jac .*= cb.con_scale
    _treat_fixed_variable_jac_dense!(cb.fixed_handler, cb, jac)
end
function _treat_fixed_variable_jac_dense!(fixed_handler::RelaxBound, cb::DenseCallback, jac) end
function _treat_fixed_variable_jac_dense!(fixed_handler::MakeParameter, cb::DenseCallback{T}, jac) where T
    jac[:,fixed_handler.fixed] .= zero(T)
end


function _eval_lag_hess_wrapper!(
    cb::DenseCallback{T},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractMatrix;
    obj_weight = one(T)
    ) where T
    
    hess_dense!(
        cb.nlp, x, y, hess;
        obj_weight=obj_weight * cb.obj_scale[]
    )
    
    _treat_fixed_variable_lag_hess_dense!(cb.fixed_handler, cb, hess)
end
function _treat_fixed_variable_lag_hess_dense!(fixed_handler::RelaxBound, cb::DenseCallback, hess) end
function _treat_fixed_variable_lag_hess_dense!(fixed_handler::MakeParameter, cb::DenseCallback{T}, hess) where T
    fixed = fixed_handler.fixed
    hess[:,fixed] .= zero(T)
    hess[fixed,:] .= zero(T)
    @inbounds @simd for i in fixed
        hess[i,i] = one(T)
    end
end



function update_z!(cb, zl, zu, jacl)
    _update_z!(cb.fixed_handler, zl, zu, jacl, get_minimize(cb.nlp) ? 1 : -1)
end

function _update_z!(fixed_handler::MakeParameter, zl, zu, jacl, sense) 
    zl_r = @view(zl[fixed_handler.fixed])
    zu_r = @view(zu[fixed_handler.fixed])
    map!(
        (x,y)->sense * max(x+y,0),
        zl_r,
        fixed_handler.grad_storage,
        jacl
    )
    map!(
        (x,y)->sense * max(-(x+y),0),
        zu_r,
        fixed_handler.grad_storage,
        jacl,
    )
end
function _update_z!(fixed_handler::RelaxBound, zl, zu, jacl, sense) end
