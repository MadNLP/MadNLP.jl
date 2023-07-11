abstract type AbstractFixedVariableHandler end
struct MakeParameter{VI} <: AbstractFixedVariableHandler
    fixed::VI
    fixedj::VI
    fixedh::VI
end
struct RelaxBound <: AbstractFixedVariableHandler end

struct NLPModelWrapper{
    T, VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    I <: AbstractNLPModel{T, VT},
    FH <: AbstractFixedVariableHandler
    } <: AbstractNLPModel{T, VT}
    
    inner::I

    l_buffer::VT

    jac_I::VI
    jac_J::VI
    hess_I::VI
    hess_J::VI

    obj_scale::T
    con_scale::VT
    jac_scale::VT

    fixed_handler::FH
    
    meta::NLPModelMeta{T, VT}
    counters::NLPModels.Counters
end

get_obj_scale(f::AbstractVector{T},max_gradient) where T = min(one(T), max_gradient / norm(f,Inf))
function get_con_scale(jac_I,jac_buffer::VT, ncon, nnzj, max_gradient) where {T, VT <: AbstractVector{T}}
    
    con_scale = fill!(similar(jac_buffer, ncon), 1)
    jac_scale = fill!(similar(jac_buffer, nnzj), 1)    
    @inbounds for i in 1:nnzj
        row = jac_I[i]
        con_scale[row] = max(con_scale[row], abs(jac_buffer[i]))
    end
    @inbounds for i in eachindex(con_scale)
        con_scale[i] = min(one(T), max_gradient / con_scale[i])
    end
    @inbounds for i in 1:nnzj
        index = jac_I[i]
        jac_scale[i] = con_scale[index]
    end
    return con_scale, jac_scale
end

function create_fixed_handler(
    fixed_variable_treatment::Type{MakeParameter},
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

    fill!(@view(lvar[fixed]), -Inf)
    fill!(@view(uvar[fixed]),  Inf)

    nnzh = nnzh + nfixed

    fixed_handler = MakeParameter(fixed,fixedj,fixedh)

    
    return fixed_handler, nnzj, nnzh
end

function create_fixed_handler(
    fixed_variable_treatment::Type{RelaxBound},
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
    
    lvar[fixed] .+= opt.boudn_relax_factor
    uvar[fixed] .+= opt.boudn_relax_factor

    fixed_handler = RelaxBound()

    
    return fixed_handler, nnzj, nnzh
end

function NLPModelWrapper(
    nlp::AbstractNLPModel{T, VT};
    opt,
    ) where {T, VT}

    n = get_nvar(nlp)
    m = get_ncon(nlp)
    nnzj = get_nnzj(nlp)
    nnzh = get_nnzh(nlp)

    x0    = copy(get_x0(nlp))
    lvar = copy(nlp.meta.lvar)
    uvar = copy(nlp.meta.uvar)
    
    l_buffer = similar(x0, m)
    grad_buffer = similar(x0, n)
    jac_buffer = similar(x0, nnzj)
    hess_buffer = similar(x0, nnzh)

    jac_I = similar(x0, Int, nnzj)
    jac_J = similar(x0, Int, nnzj)
    hess_I = similar(x0, Int, nnzh)
    hess_J = similar(x0, Int, nnzh)

    NLPModels.jac_structure!(nlp,jac_I,jac_J)
    NLPModels.hess_structure!(nlp,hess_I,hess_J)

    fixed_handler, nnzj, nnzh = create_fixed_handler(
        opt.fixed_variable_treatment,
        lvar, uvar, nnzj, nnzh,
        jac_I, jac_J, hess_I, hess_J;
        opt = opt
    )

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
    NLPModels.jac_coord!(nlp,x0,jac_buffer)
    NLPModels.grad!(nlp,x0,grad_buffer)

    con_scale, jac_scale = get_con_scale(jac_I, jac_buffer, get_ncon(nlp), get_nnzj(nlp), opt.nlp_scaling_max_gradient)
    obj_scale = get_obj_scale(grad_buffer, opt.nlp_scaling_max_gradient)

    l_buffer .*= con_scale
    y0   = get_y0(nlp) ./ con_scale
    lcon = get_lcon(nlp) .* con_scale
    ucon = get_ucon(nlp) .* con_scale


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
    
    return NLPModelWrapper(
        nlp,
        l_buffer,
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
    copyto!(view(I, 1:length(nlp.hess_I)), nlp.hess_I)
    copyto!(view(J, 1:length(nlp.hess_I)), nlp.hess_J)

    _treat_fixed_variable_hess_structure!(nlp.fixed_handler, nlp, I, J)
end

function _treat_fixed_variable_hess_structure!(fixed_handler::RelaxBound, nlp, I, J) end
function _treat_fixed_variable_hess_structure!(fixed_handler::MakeParameter, nlp, I, J)
    fixed = fixed_handler.fixed
    nfixed = length(fixed)
    copyto!(@view(I[length(nlp.hess_I)+1:end]), fixed)
    copyto!(@view(J[length(nlp.hess_J)+1:end]), fixed)
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
    grad .*= nlp.obj_scale
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
    return NLPModels.obj(nlp.inner,x)* nlp.obj_scale
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
        obj_weight=obj_weight * nlp.obj_scale
    )
    _treat_fixed_variable_hess_coord!(nlp.fixed_handler, nlp, hess) 
end

function _treat_fixed_variable_hess_coord!(fixed_handler::RelaxBound, nlp, hess) end
function _treat_fixed_variable_hess_coord!(fixed_handler::MakeParameter, nlp::NLPModelWrapper{T}, hess) where T
    nnzh_orig = nlp.inner.meta.nnzh
    fill!(@view(hess[fixed_handler.fixedh]), zero(T))
    fill!(@view(hess[nnzh_orig+1:end]), one(T))
end
