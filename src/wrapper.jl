#= 

NLPModelWrapper wraps an NLP model and do several manipulations
- introducing slack variables for inequalities
- fixed variable treatment (via elimination)
- compression
- scaling
- consraint rhs

=#
mutable struct NLPModelWrapper{T, VT <: AbstractVector{T}, VI <: AbstractVector{Int}} <: AbstractNLPModel{T, VT}
    inner::AbstractNLPModel{T, VT}

    x_buffer::VT
    cons_buffer::VT
    grad_buffer::VT
    jac_buffer::VT
    hess_buffer::VT
    rhs::VT

    ind_free::VI
    ind_free_j::VI
    ind_free_h::VI
    ind_cons::VI
    ind_jac_slack::VI

    var_orig_range::UnitRange{Int}
    var_slack_range::UnitRange{Int}
    con_slack_range::UnitRange{Int}

    jac_I::VI
    jac_J::VI
    hess_I::VI
    hess_J::VI

    hess_order::VI
    jac_order::VI
    hess_ptr::VI
    jac_ptr::VI
    
    obj_scale::T
    con_scale::VT
    jac_scale::VT
    
    meta::NLPModelMeta{T, VT}
    counters::NLPModels.Counters
end

function NLPModelWrapper(nlp::AbstractNLPModel{T, VT}) where {T, VT}

    n = get_nvar(nlp)
    m = get_ncon(nlp)
    nnzj = get_nnzj(nlp)
    nnzh = get_nnzh(nlp)
    
    x_buffer    = copy(get_x0(nlp))

    isfree = (nlp.meta.lvar .!= nlp.meta.uvar)
    isineq = (nlp.meta.lcon .!= nlp.meta.ucon)    
    ind_free = findall(isfree)
    ind_ineq = findall(isineq)
    ind_eq = findall(x->!x, isineq)

    n0 = length(ind_free)
    n_slack = length(ind_ineq)
    nn = length(ind_free) + n_slack
    
    var_orig_range = 1:n0
    var_slack_range = n0+1:nn
    con_slack_range = m-n_slack+1:m

    ind_cons = similar(x_buffer, Int, m)
    ind_cons[ind_eq] .= 1:m-n_slack
    ind_cons[ind_ineq] .= con_slack_range

    cons_buffer = similar(x_buffer, m)
    grad_buffer = similar(x_buffer, n)
    jac_buffer = similar(x_buffer, nnzj)
    hess_buffer = similar(x_buffer, nnzh)

    map = cumsum(isfree)

    jac_I = similar(x_buffer, Int, nnzj)
    jac_J = similar(x_buffer, Int, nnzj)
    hess_I = similar(x_buffer, Int, nnzh)
    hess_J = similar(x_buffer, Int, nnzh)

    jac_structure!(nlp,jac_I,jac_J)
    hess_structure!(nlp,hess_I,hess_J)

    # scale constraints
    max_gradient = 100
    
    NLPModels.cons!(nlp,x_buffer,cons_buffer)
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
    cons_buffer .*= con_scale

    # scale objective
    NLPModels.grad!(nlp,x_buffer,grad_buffer)
    obj_scale = min(one(T), max_gradient / norm(grad_buffer,Inf))


    ind_free_j = findall(j->isfree[j], jac_J)
    ind_free_h = findall(@view(isfree[hess_I]) .& @view(isfree[hess_J]))

    y0   = similar(x_buffer, m) .* con_scale
    lcon = similar(x_buffer, m) .* con_scale
    ucon = similar(x_buffer, m) .* con_scale
    
    copyto!(@view(y0[ind_cons])   , nlp.meta.y0)
    copyto!(@view(lcon[ind_cons]) , nlp.meta.lcon)
    copyto!(@view(ucon[ind_cons]) , nlp.meta.ucon)

    lcon_slk = @view lcon[con_slack_range]
    ucon_slk = @view ucon[con_slack_range]
    x0   = [@view nlp.meta.x0[ind_free]; @view cons_buffer[ind_ineq]]
    lvar = [@view nlp.meta.lvar[ind_free]; lcon_slk]
    uvar = [@view nlp.meta.uvar[ind_free]; ucon_slk] 
    
    fill!(@view(lcon[con_slack_range]), zero(T))
    fill!(@view(ucon[con_slack_range]), zero(T))

    hess_I = map[@view hess_I[ind_free_h]]
    hess_J = map[@view hess_J[ind_free_h]]
    jac_I  = [@view ind_cons[@view jac_I[ind_free_j]]; con_slack_range]
    jac_J  = [@view map[@view jac_J[ind_free_j]]; var_slack_range]
    
    hess_order, hess_ptr = sortcoo!(hess_I, hess_J)

    idx = similar(hess_order,Int)
    idx[hess_order] .= 1:length(idx)

    hess_ptrmap = ptrmap(hess_ptr)
    hess_order = hess_ptrmap[idx]

    jac_order, jac_ptr = sortcoo!(jac_I, jac_J)
    jac_ptrmap = ptrmap(jac_ptr)

    idx = similar(jac_order,Int)
    idx[jac_order] .= 1:length(idx)

    ind_jac_slack = jac_ptrmap[@view idx[nnzj+1:nnzj+n_slack]]
    jac_order = jac_ptrmap[@view idx[1:length(ind_free_j)]]

    rhs = [@view lcon[1:m-n_slack]; fill!(similar(x_buffer, n_slack), zero(T))]

    return NLPModelWrapper(
        nlp,
        x_buffer,
        cons_buffer,
        grad_buffer,
        jac_buffer,
        hess_buffer,
        rhs,
        ind_free,
        ind_free_j,
        ind_free_h,
        ind_cons,
        ind_jac_slack,
        var_orig_range,
        var_slack_range,
        con_slack_range,
        jac_I,
        jac_J,
        hess_I,
        hess_J,
        hess_order,
        jac_order,
        hess_ptr,
        jac_ptr,
        obj_scale,
        con_scale,
        jac_scale,
        NLPModelMeta(
            nn,
            x0 = x0,
            lvar = lvar,
            uvar = uvar,
            ncon = m,
            y0 = y0,
            lcon = lcon,
            ucon = ucon,
            nnzj = length(jac_ptr)-1,
            nnzh = length(hess_ptr)-1,
            minimize = nlp.meta.minimize
        ),
        NLPModels.Counters()
    )
end

function ptrmap(ptr)
    map = similar(ptr,ptr[end]-1)
    for i=1:length(ptr)-1
        map[ptr[i]:ptr[i+1]-1] .= i
    end
    return map
end


function NLPModels.jac_structure!(nlp::NLPModelWrapper,I::AbstractVector,J::AbstractVector)
    copyto!(I, view(nlp.jac_I, @view(nlp.jac_ptr[1:end-1])))
    copyto!(J, view(nlp.jac_J, @view(nlp.jac_ptr[1:end-1])))
end


function NLPModels.hess_structure!(nlp::NLPModelWrapper,I::AbstractVector,J::AbstractVector)
    copyto!(I, view(nlp.hess_I, @view(nlp.hess_ptr[1:end-1])))
    copyto!(J, view(nlp.hess_J, @view(nlp.hess_ptr[1:end-1])))
end


function NLPModels.cons!(nlp::NLPModelWrapper,x::AbstractVector,c::AbstractVector)
    copyto!(view(nlp.x_buffer, nlp.ind_free), view(x, nlp.var_orig_range))
    NLPModels.cons!(nlp.inner, nlp.x_buffer, nlp.cons_buffer)
    nlp.cons_buffer .*= nlp.con_scale
    c[nlp.ind_cons] .= nlp.cons_buffer
    view(c,nlp.con_slack_range) .-= view(x,nlp.var_slack_range)
    c .-= nlp.rhs
    return c
end


function NLPModels.jac_coord!(nlp::NLPModelWrapper,x::AbstractVector,jac::AbstractVector)
    copyto!(view(nlp.x_buffer, nlp.ind_free), view(x, nlp.var_orig_range))
    NLPModels.jac_coord!(nlp.inner, nlp.x_buffer, nlp.jac_buffer)
    nlp.jac_buffer .*= nlp.jac_scale
    fill!(jac, 0)
    add!(jac,nlp.jac_buffer,nlp.jac_order,nlp.ind_free_j)
    fill!(view(jac, nlp.ind_jac_slack), -1)
end

function NLPModels.grad!(nlp::NLPModelWrapper,x::AbstractVector,grad::AbstractVector)
    copyto!(view(nlp.x_buffer, nlp.ind_free), view(x, nlp.var_orig_range))
    NLPModels.grad!(nlp.inner, nlp.x_buffer, nlp.grad_buffer)
    grad[nlp.var_orig_range] .= view(nlp.grad_buffer, nlp.ind_free).* nlp.obj_scale
    fill!(view(grad,nlp.var_slack_range), 0)
end

function NLPModels.obj(nlp::NLPModelWrapper,x::AbstractVector)
    copyto!(view(nlp.x_buffer, nlp.ind_free), view(x, nlp.var_orig_range))
    return NLPModels.obj(nlp.inner,nlp.x_buffer)* nlp.obj_scale
end

function NLPModels.hess_coord!(
    nlp::NLPModelWrapper{T},x::AbstractVector,y::AbstractVector,hess::AbstractVector;
    obj_weight = 1.) where T

    fill!(hess, zero(T))
    nlp.cons_buffer .= view(y, nlp.ind_cons) .* nlp.con_scale
    copyto!(view(nlp.x_buffer, nlp.ind_free), view(x, nlp.var_orig_range))
    NLPModels.hess_coord!(nlp.inner, nlp.x_buffer, nlp.cons_buffer, nlp.hess_buffer; obj_weight=obj_weight * nlp.obj_scale)
    add!(hess,nlp.hess_buffer,nlp.hess_order,nlp.ind_free_h)
end

function add!(dst,src,idx1,idx2)
    @inbounds @simd for i in 1:length(idx1)
        dst[idx1[i]] += src[idx2[i]]
    end
end

sortcoo!(I,J) = (
    sort!(TupleVector(I,J); by = x -> (x[2],x[1]), alg= Base.Sort.MergeSort).V,
    getptr(I,J)
)

function getptr(I,J)
    ptr = similar(I,Int,length(I)+1)

    cnt = 1
    ptr[1] = 1
    prev = (I[1],J[1])
    @inbounds for i in 2:length(I)
        curr = (I[i],J[i])
        
        if prev != curr
            ptr[cnt += 1] = i
        end

        prev = curr
    end
    ptr[cnt += 1] = length(I) +1

    resize!(ptr,cnt)
end

struct TupleVector{T} <: AbstractVector{Tuple{T,T,T}}
    I::Vector{T}
    J::Vector{T}
    V::Vector{T}
end

function TupleVector(I,J)
    n = length(I)
    @assert n == length(J)
    V = (similar(I,Int) .= 1:n)
    return TupleVector(I,J,V)
end

Base.getindex(c::TupleVector{T}, i) where T = (c.I[i], c.J[i], c.V[i])
function Base.setindex!(c::TupleVector{T}, tup, i) where T
    c.I[i] = tup[1]
    c.J[i] = tup[2]
    c.V[i] = tup[3]
end
Base.size(c::TupleVector{T}) where T = (length(c.I),)
