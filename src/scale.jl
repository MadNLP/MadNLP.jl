struct FixedVariableMakeParameter{VI}
    fixed::VI
    fixedj::VI
    fixedh::VI
end

struct ScaledNLPModel{
    T, VT <: AbstractVector{T},
    VI <: AbstractVector{Int}, I <: AbstractNLPModel{T, VT},
    FH
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

function ScaledNLPModel(nlp::AbstractNLPModel{T, VT}; max_gradient=100) where {T, VT}

    n = get_nvar(nlp)
    m = get_ncon(nlp)
    nnzj = get_nnzj(nlp)
    nnzh = get_nnzh(nlp)

    x_buffer    = copy(get_x0(nlp))
    l_buffer = similar(x_buffer, m)
    grad_buffer = similar(x_buffer, n)
    jac_buffer = similar(x_buffer, nnzj)
    hess_buffer = similar(x_buffer, nnzh)

    jac_I = similar(x_buffer, Int, nnzj)
    jac_J = similar(x_buffer, Int, nnzj)
    hess_I = similar(x_buffer, Int, nnzh)
    hess_J = similar(x_buffer, Int, nnzh)

    jac_structure!(nlp,jac_I,jac_J)
    hess_structure!(nlp,hess_I,hess_J)

    isfixed  = (nlp.meta.lvar .== nlp.meta.uvar)
    isfixedj = isfixed[jac_J]
    isfixedh = isfixed[hess_I] .|| isfixed[hess_J]
    fixed  = findall(isfixed)
    fixedj = findall(isfixedj)
    fixedh = findall(isfixedh)
    nfixed = length(fixed)
    fixed_handler = FixedVariableMakeParameter(fixed, fixedj, fixedh)

    # scale constraints
    NLPModels.cons!(nlp,x_buffer,l_buffer)
    NLPModels.jac_coord!(nlp,x_buffer,jac_buffer)
    
    # TODO: get max_gradient from option
    con_scale, jac_scale = get_con_scale(jac_I, jac_buffer, get_ncon(nlp), get_nnzj(nlp), max_gradient)

    # scale objective
    NLPModels.grad!(nlp,x_buffer,grad_buffer)
    obj_scale = min(one(T), max_gradient / norm(grad_buffer,Inf))

    y0   = get_y0(nlp) .* con_scale
    lcon = get_lcon(nlp) .* con_scale
    ucon = get_ucon(nlp) .* con_scale

    lvar = copy(get_lvar(nlp))
    uvar = copy(get_uvar(nlp))
    fill!(@view(lvar[fixed]), -Inf)
    fill!(@view(uvar[fixed]),  Inf)
    

    return ScaledNLPModel(
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
            x0 = get_x0(nlp),
            lvar = lvar,
            uvar = uvar,
            ncon = m,
            y0 = y0,
            lcon = lcon,
            ucon = ucon,
            nnzj = nnzj,
            nnzh = nnzh + nfixed,
            minimize = nlp.meta.minimize
        ),
        NLPModels.Counters()
    )
end

function NLPModels.jac_structure!(nlp::ScaledNLPModel,I::AbstractVector,J::AbstractVector)
    copyto!(I, nlp.jac_I)
    copyto!(J, nlp.jac_J)
end


function NLPModels.hess_structure!(nlp::ScaledNLPModel,I::AbstractVector,J::AbstractVector)
    copyto!(view(I, 1:length(nlp.hess_I)), nlp.hess_I)
    copyto!(view(J, 1:length(nlp.hess_I)), nlp.hess_J)

    fixed = nlp.fixed_handler.fixed
    nfixed = length(fixed)
    copyto!(@view(I[length(nlp.hess_I)+1:end]), fixed)
    copyto!(@view(J[length(nlp.hess_J)+1:end]), fixed)
end


function NLPModels.cons!(nlp::ScaledNLPModel,x::AbstractVector,c::AbstractVector)
    NLPModels.cons!(nlp.inner, x,c)
    c .*= nlp.con_scale
    return c
end


function NLPModels.jac_coord!(nlp::ScaledNLPModel{T},x::AbstractVector,jac::AbstractVector) where T
    nnzj_orig = nlp.inner.meta.nnzj
    NLPModels.jac_coord!(nlp.inner, x, jac)
    jac .*= nlp.jac_scale
    fill!(@view(jac[nlp.fixed_handler.fixedj]), zero(T))
end

function NLPModels.grad!(nlp::ScaledNLPModel{T},x::AbstractVector,grad::AbstractVector) where T
    NLPModels.grad!(nlp.inner, x, grad)
    grad .*= nlp.obj_scale
    map!(
        -,
        @view(grad[nlp.fixed_handler.fixed]),
        @view(nlp.inner.meta.lvar[nlp.fixed_handler.fixed])
    )
end

function NLPModels.obj(nlp::ScaledNLPModel,x::AbstractVector)
    return NLPModels.obj(nlp.inner,x)* nlp.obj_scale
end

function NLPModels.hess_coord!(
    nlp::ScaledNLPModel{T},x::AbstractVector,y::AbstractVector,hess::AbstractVector; 
    obj_weight = 1.) where T

    nnzh_orig = nlp.inner.meta.nnzh
    
    nlp.l_buffer .= y .* nlp.con_scale
    NLPModels.hess_coord!(
        nlp.inner, x, nlp.l_buffer, view(hess, 1:nnzh_orig);
        obj_weight=obj_weight * nlp.obj_scale
    )
    fill!(@view(hess[nlp.fixed_handler.fixedh]), zero(T))
    fill!(@view(hess[nnzh_orig+1:end]), one(T))
end
