struct SparseCallback end
struct DenseCallback end

abstract type AbstractFixedVariableHandler end
struct MakeParameter{VI} <: AbstractFixedVariableHandler
    fixed::VI
    fixedj::VI
    fixedh::VI
end
struct RelaxBound <: AbstractFixedVariableHandler end

struct NLPModelWrapper{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    I <: AbstractNLPModel{T, VT},
    FH <: AbstractFixedVariableHandler,
    } <: AbstractNLPModel{T, VT}
    
    inner::I

    l_buffer::VT
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
    
    meta::NLPModelMeta{T, VT}
    counters::NLPModels.Counters
end

function set_obj_scale!(obj_scale, f::VT,max_gradient) where {T, VT <: AbstractVector{T}}
    obj_scale[] = min(one(T), max_gradient / norm(f,Inf))
end
function set_con_scale!(con_scale::VT, jac_scale, jac_I,jac_buffer, max_gradient) where {T, VT <: AbstractVector{T}}
    fill!(con_scale, one(T))
    @inbounds @simd for i in 1:length(jac_I)
        row = jac_I[i]
        con_scale[row] = max(con_scale[row], abs(jac_buffer[i]))
    end
    @inbounds @simd for i in eachindex(con_scale)
        con_scale[i] = min(one(T), max_gradient / con_scale[i])
    end
end

function set_jac_scale!(jac_scale::VT, con_scale, jac_I) where {T, VT <: AbstractVector{T}}
    fill!(jac_scale, one(T))
    @inbounds @simd for i in 1:length(jac_I)
        index = jac_I[i]
        jac_scale[i] = con_scale[index]
    end
end

function create_fixed_handler(
    fixed_variable_treatment::Type{MakeParameter},
    x0,
    lvar,
    uvar,
    nnzj,
    nnzh,
    jac_I,
    jac_J,
    hess_I,
    hess_J;
    opt
    )
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

    fixed_handler = MakeParameter(fixed,fixedj,fixedh)

    
    return fixed_handler, nnzj, nnzh
end

function create_fixed_handler(
    fixed_variable_treatment::Type{RelaxBound},
    x0,
    lvar,
    uvar,
    nnzj,
    nnzh,
    jac_I,
    jac_J,
    hess_I,
    hess_J;
    opt
    )

    fixed_handler = RelaxBound()

    
    return fixed_handler, nnzj, nnzh
end

function create_model_wrapper(
    ::Type{SparseCallback},
    nlp::AbstractNLPModel{T, VT},
    opt,
    ) where {T, VT}

    n = get_nvar(nlp)
    m = get_ncon(nlp)
    nnzj = get_nnzj(nlp)
    nnzh = get_nnzh(nlp)

    x0   = similar(get_x0(nlp))
    lvar = similar(nlp.meta.lvar)
    uvar = similar(nlp.meta.uvar)
    y0   = similar(get_y0(nlp)) 
    lcon = similar(get_lcon(nlp)) 
    ucon = similar(get_ucon(nlp))

    l_buffer = similar(x0, m)
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

    fixed_handler, nnzj, nnzh = create_fixed_handler(
        opt.fixed_variable_treatment,
        get_x0(nlp), get_lvar(nlp), get_uvar(nlp), nnzj, nnzh,
        jac_I, jac_J, hess_I, hess_J;
        opt = opt
    )

    return NLPModelWrapper(
        nlp,
        l_buffer,
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
        fixed_handler,
        NLPModelMeta(
            n,
            x0 = x0,
            lvar = lvar,
            uvar = uvar,
            ncon = m,
            y0 = y0,
            lcon = lcon,
            ucon = ucon,
            nnzj = nnzj,
            nnzh = nnzh,
            minimize = nlp.meta.minimize
        ),
        NLPModels.Counters()
    )
end

function _treat_fixed_variable_initialize!(fixed_handler::RelaxBound, x0, lvar, uvar) end
function _treat_fixed_variable_initialize!(fixed_handler::MakeParameter, x0, lvar, uvar) where T
    fixed = fixed_handler.fixed
    copyto!(@view(x0[fixed]), @view(lvar[fixed]))
    fill!(@view(lvar[fixed]), -Inf)
    fill!(@view(uvar[fixed]),  Inf)
end
function initialize!(
    wrapper::NLPModelWrapper,
    x, xl, xu, y0, rhs,
    ind_ineq,
    opt
    )

    x0= variable(x)
    lvar= variable(xl)
    uvar= variable(xu)
    
    fixed_handler = wrapper.fixed_handler
    nlp = wrapper.inner
    
    l_buffer =wrapper.l_buffer
    jac_buffer =wrapper.jac_buffer
    grad_buffer =wrapper.grad_buffer

    lcon = nlp.meta.lcon
    ucon = nlp.meta.ucon
    
    jac_I = wrapper.jac_I
    jac_J = wrapper.jac_J

    x0   .= get_x0(nlp)
    y0   .= get_y0(nlp) 
    lvar .= nlp.meta.lvar
    uvar .= nlp.meta.uvar  
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
    
    NLPModels.cons!(nlp,x0,l_buffer)
    is = findall(.!(lcon .== -Inf .&& ucon .== Inf) .&& (lcon .!= ucon))
    set_initial_bounds!(
        @view(lcon[is]),
        @view(ucon[is]),
        opt.tol
    )
    initialize_variables!(
        @view(l_buffer[is]),
        @view(lcon[is]),
        @view(ucon[is]),
        opt.bound_push,
        opt.bound_fac
    )
    
    slack(xl) .= view(lcon, ind_ineq)
    slack(xu) .= view(ucon, ind_ineq)
    rhs .= (lcon.==ucon).*lcon
    copyto!(slack(x), @view(l_buffer[ind_ineq]))
end

function set_scaling!(wrapper, x, y0, rhs, nlp_scaling_max_gradient)
    x0= variable(x)

    nlp = wrapper.inner
    lcon = nlp.meta.lcon
    ucon = nlp.meta.ucon
    obj_scale = wrapper.obj_scale
    con_scale = wrapper.con_scale
    jac_scale = wrapper.jac_scale    
    l_buffer =wrapper.l_buffer
    jac_buffer =wrapper.jac_buffer
    grad_buffer =wrapper.grad_buffer

    # Set scaling
    NLPModels.jac_coord!(nlp,x0,jac_buffer)
    set_con_scale!(con_scale, jac_scale, wrapper.jac_I, jac_buffer, nlp_scaling_max_gradient)
    set_jac_scale!(jac_scale, con_scale, wrapper.jac_I)
    
    NLPModels.grad!(nlp,x0,grad_buffer)
    set_obj_scale!(obj_scale, grad_buffer, nlp_scaling_max_gradient)

    y0 ./= con_scale
    rhs  .*= con_scale
end

function NLPModels.jac_structure!(
    nlp::NLPModelWrapper,
    I::AbstractVector,J::AbstractVector
    )
    
    copyto!(I, nlp.jac_I)
    copyto!(J, nlp.jac_J)
end

function NLPModels.hess_structure!(
    nlp::NLPModelWrapper,
    I::AbstractVector,J::AbstractVector
    ) 
    copyto!(I, nlp.hess_I)
    copyto!(J, nlp.hess_J)
end


function NLPModels.cons!(nlp::NLPModelWrapper,x::AbstractVector,c::AbstractVector)
    NLPModels.cons!(nlp.inner, x,c)
    c .*= nlp.con_scale
    return c
end


function NLPModels.jac_coord!(
    nlp::NLPModelWrapper,
    x::AbstractVector,
    jac::AbstractVector
    )
    
    nnzj_orig = nlp.inner.meta.nnzj
    NLPModels.jac_coord!(nlp.inner, x, jac)
    jac .*= nlp.jac_scale

    _treat_fixed_variable_jac_coord!(nlp.fixed_handler, nlp, x, jac)
end
function _treat_fixed_variable_jac_coord!(fixed_handler::RelaxBound, nlp, x, jac) end
function _treat_fixed_variable_jac_coord!(fixed_handler::MakeParameter, nlp::NLPModelWrapper{T}, x, jac) where T
    fill!(@view(jac[fixed_handler.fixedj]), zero(T))
end

function NLPModels.grad!(
    nlp::NLPModelWrapper{T},
    x::AbstractVector,
    grad::AbstractVector
    ) where T
    
    NLPModels.grad!(nlp.inner, x, grad)
    grad .*= nlp.obj_scale[]
    _treat_fixed_variable_grad!(nlp.fixed_handler, nlp, x, grad)
end
function _treat_fixed_variable_grad!(fixed_handler::RelaxBound, nlp, x, grad) end
function _treat_fixed_variable_grad!(fixed_handler::MakeParameter, nlp::NLPModelWrapper{T}, x, grad) where T
    map!(
        (x,y)->x-y,
        @view(grad[fixed_handler.fixed]),
        @view(x[nlp.fixed_handler.fixed]),
        @view(nlp.inner.meta.lvar[nlp.fixed_handler.fixed])
    )
end

function NLPModels.obj(nlp::NLPModelWrapper,x::AbstractVector)
    return NLPModels.obj(nlp.inner,x)* nlp.obj_scale[]
end

function NLPModels.hess_coord!(
    nlp::NLPModelWrapper{T},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector; 
    obj_weight = 1.0
    ) where T
    
    nnzh_orig = nlp.inner.meta.nnzh
    
    nlp.l_buffer .= y .* nlp.con_scale
    NLPModels.hess_coord!(
        nlp.inner, x, nlp.l_buffer, view(hess, 1:nnzh_orig);
        obj_weight=obj_weight * nlp.obj_scale[]
    )
    _treat_fixed_variable_hess_coord!(nlp.fixed_handler, nlp, hess) 
end

function _treat_fixed_variable_hess_coord!(fixed_handler::RelaxBound, nlp, hess) end
function _treat_fixed_variable_hess_coord!(fixed_handler::MakeParameter, nlp::NLPModelWrapper{T}, hess) where T
    nnzh_orig = nlp.inner.meta.nnzh
    fill!(@view(hess[fixed_handler.fixedh]), zero(T))
    fill!(@view(hess[nnzh_orig+1:end]), one(T))
end

function jac_dense!(
    nlp::NLPModelWrapper{T},
    x::AbstractVector,
    jac
    ) where T
    jac_buffer = nlp.jac_buffer
    NLPModels.jac_coord!(nlp, x, jac_buffer)
    fill!(jac, zero(T))
    @inbounds @simd for k=1:length(nlp.jac_I)
        i,j = nlp.jac_I[k], nlp.jac_J[k]
        jac[i,j] += jac_buffer[k]
    end
end

function hess_dense!(
    nlp::NLPModelWrapper{T},
    x::AbstractVector,
    y::AbstractVector,
    hess;
    obj_weight = one(T)
    ) where T
    
    hess_buffer = nlp.hess_buffer
    NLPModels.hess_coord!(nlp, x, y, hess_buffer; obj_weight=obj_weight)
    fill!(hess, zero(T))
    @inbounds @simd for k=1:length(nlp.hess_I)
        i,j = nlp.hess_I[k], nlp.hess_J[k]
        hess[i,j] += hess_buffer[k]
    end
    _treat_fixed_variable_hess_dense!(nlp.fixed_handler, nlp, hess)
end
function _treat_fixed_variable_hess_dense!(fixed_handler::RelaxBound, nlp, hess) end
function _treat_fixed_variable_hess_dense!(fixed_handler::MakeParameter, nlp::NLPModelWrapper{T}, hess) where T
    nnzh_orig = nlp.inner.meta.nnzh
    
    for i in fixed_handler.fixed
        hess[i,i] = one(T)
    end
end
