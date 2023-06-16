mutable struct ScaledNLPModel{T, VT <: AbstractVector{T}, VI <: AbstractVector{Int}} <: AbstractNLPModel{T, VT}
    inner::AbstractNLPModel{T, VT}

    l_buffer::VT

    jac_I::VI
    jac_J::VI
    hess_I::VI
    hess_J::VI

    obj_scale::T
    con_scale::VT
    jac_scale::VT
    
    meta::NLPModelMeta{T, VT}
    counters::NLPModels.Counters
end

function ScaledNLPModel(nlp::AbstractNLPModel{T, VT}) where {T, VT}

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

    # scale constraints
    max_gradient = 100
    
    NLPModels.cons!(nlp,x_buffer,l_buffer)
    NLPModels.jac_coord!(nlp,x_buffer,jac_buffer)
    con_scale = fill!(similar(x_buffer, get_ncon(nlp)), 1)
    jac_scale = fill!(similar(x_buffer, get_nnzj(nlp)), 1)    
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

    # scale objective
    NLPModels.grad!(nlp,x_buffer,grad_buffer)
    obj_scale = min(one(T), max_gradient / norm(grad_buffer,Inf))

    y0   = get_y0(nlp) .* con_scale
    lcon = get_lcon(nlp) .* con_scale
    ucon = get_ucon(nlp) .* con_scale


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
        NLPModelMeta(
            n,
            x0 = get_x0(nlp),
            lvar = get_lvar(nlp),
            uvar = get_uvar(nlp),
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

function NLPModels.jac_structure!(nlp::ScaledNLPModel,I::AbstractVector,J::AbstractVector)
    copyto!(I, nlp.jac_I)
    copyto!(J, nlp.jac_J)
end


function NLPModels.hess_structure!(nlp::ScaledNLPModel,I::AbstractVector,J::AbstractVector)
    copyto!(I, nlp.hess_I)
    copyto!(J, nlp.hess_J)
end


function NLPModels.cons!(nlp::ScaledNLPModel,x::AbstractVector,c::AbstractVector)
    NLPModels.cons!(nlp.inner, x,c)
    c .*= nlp.con_scale
    return c
end


function NLPModels.jac_coord!(nlp::ScaledNLPModel,x::AbstractVector,jac::AbstractVector)
    NLPModels.jac_coord!(nlp.inner, x, jac)
    jac .*= nlp.jac_scale
end

function NLPModels.grad!(nlp::ScaledNLPModel,x::AbstractVector,grad::AbstractVector)
    NLPModels.grad!(nlp.inner, x, grad)
    grad .*= nlp.obj_scale
end

function NLPModels.obj(nlp::ScaledNLPModel,x::AbstractVector)
    return NLPModels.obj(nlp.inner,x)* nlp.obj_scale
end

function NLPModels.hess_coord!(
    nlp::ScaledNLPModel{T},x::AbstractVector,y::AbstractVector,hess::AbstractVector; 
    obj_weight = 1.) where T

    nlp.l_buffer .= y .* nlp.con_scale
    NLPModels.hess_coord!(nlp.inner, x, nlp.l_buffer, hess; obj_weight=obj_weight * nlp.obj_scale)
end
