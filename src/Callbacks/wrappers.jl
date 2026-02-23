abstract type AbstractWrapperModel{T,VT} <: NLPModels.AbstractNLPModel{T,VT} end

struct DenseWrapperModel{T,VT,T2,VT2,MT2, I <: NLPModels.AbstractNLPModel{T2,VT2}} <: AbstractWrapperModel{T,VT}
    inner::I
    x::VT2
    y::VT2
    con::VT2
    grad::VT2
    jac::MT2
    hess::MT2
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
end

mutable struct SparseWrapperModel{T,VT,T2,VI2,VT2,I <: NLPModels.AbstractNLPModel{T2,VT2}} <: AbstractWrapperModel{T,VT}
    const inner::I
    const jrows::VI2
    const jcols::VI2
    const hrows::VI2
    const hcols::VI2
    const x::VT2
    const y::VT2
    const con::VT2
    const grad::VT2
    const jac::VT2
    const hess::VT2
    const meta::NLPModels.NLPModelMeta{T, VT}
    const counters::NLPModels.Counters
end

"""
    DenseWrapperModel(Arr, m)

Construct a DenseWrapperModel (a subtype of `NLPModels.AbstractNLPModel{T,typeof(Arr(m.meta.x0))}`)
from a generic NLP Model.

DenseWrapperModel can be used to interface GPU-accelerated NLP models with solvers runing on CPUs.
"""
function DenseWrapperModel(Arr, m::NLPModels.AbstractNLPModel)
    return DenseWrapperModel(
        m,
        similar(m.meta.x0, m.meta.nvar),
        similar(m.meta.x0, m.meta.ncon),
        similar(m.meta.x0, m.meta.ncon),
        similar(m.meta.x0, m.meta.nvar),
        similar(m.meta.x0, m.meta.ncon, m.meta.nvar),
        similar(m.meta.x0, m.meta.nvar, m.meta.nvar),
        NLPModels.NLPModelMeta(
            m.meta.nvar,
            x0 = Arr(m.meta.x0),
            lvar = Arr(m.meta.lvar),
            uvar = Arr(m.meta.uvar),
            ncon = m.meta.ncon,
            y0 = Arr(m.meta.y0),
            lcon = Arr(m.meta.lcon),
            ucon = Arr(m.meta.ucon),
            nnzj = m.meta.nnzj,
            nnzh = m.meta.nnzh,
            sparse_jacobian = false,
            sparse_hessian = false,
            minimize = m.meta.minimize
        ),
        NLPModels.Counters()
    )
end

"""
    SparseWrapperModel(Arr, m)

Construct a SparseWrapperModel (a subtype of `NLPModels.AbstractNLPModel{T,typeof(Arr(m.meta.x0))}`)
from a generic NLP Model.

SparseWrapperModel can be used to interface GPU-accelerated NLP models with solvers runing on CPUs.
"""
function SparseWrapperModel(Arr, m::NLPModels.AbstractNLPModel)
    return SparseWrapperModel(
        m,
        similar(m.meta.x0, Int, m.meta.nnzj),
        similar(m.meta.x0, Int, m.meta.nnzj),
        similar(m.meta.x0, Int, m.meta.nnzh),
        similar(m.meta.x0, Int, m.meta.nnzh),
        similar(m.meta.x0, m.meta.nvar),
        similar(m.meta.x0, m.meta.ncon),
        similar(m.meta.x0, m.meta.ncon),
        similar(m.meta.x0, m.meta.nvar),
        similar(m.meta.x0, m.meta.nnzj),
        similar(m.meta.x0, m.meta.nnzh),
        NLPModels.NLPModelMeta(
            m.meta.nvar,
            x0 = Arr(m.meta.x0),
            lvar = Arr(m.meta.lvar),
            uvar = Arr(m.meta.uvar),
            ncon = m.meta.ncon,
            y0 = Arr(m.meta.y0),
            lcon = Arr(m.meta.lcon),
            ucon = Arr(m.meta.ucon),
            nnzj = m.meta.nnzj,
            nnzh = m.meta.nnzh,
            sparse_jacobian = true,
            sparse_hessian = true,
            minimize = m.meta.minimize
        ),
        NLPModels.Counters()
    )
end

function NLPModels.obj(
    m::M,
    x::V
) where {M <: AbstractWrapperModel, V <: AbstractVector}
    copyto!(m.x, x)
    return NLPModels.obj(m.inner, m.x)
end

function NLPModels.cons!(
    m::M,
    x::V,
    g::V
) where {M <: AbstractWrapperModel, V <: AbstractVector}
    copyto!(m.x, x)
    NLPModels.cons!(m.inner, m.x, m.con)
    copyto!(g, m.con)
    return
end

function NLPModels.grad!(
    m::M,
    x::V,
    f::V
) where {M <: AbstractWrapperModel, V <: AbstractVector}
    copyto!(m.x, x)
    NLPModels.grad!(m.inner, m.x, m.grad)
    copyto!(f, m.grad)
    return
end

function NLPModels.jac_structure!(
    m::M,
    rows::V,
    cols::V
) where {M <: SparseWrapperModel, V <: AbstractVector}
    NLPModels.jac_structure!(m.inner, m.jrows, m.jcols)
    copyto!(rows, m.jrows)
    copyto!(cols, m.jcols)
end

function NLPModels.hess_structure!(
    m::M,
    rows::V,
    cols::V
) where {M <: SparseWrapperModel, V <: AbstractVector}
    NLPModels.hess_structure!(m.inner, m.hrows, m.hcols)
    copyto!(rows, m.hrows)
    copyto!(cols, m.hcols)
end

function NLPModels.jtprod!(
    m::SparseWrapperModel,
    x::AbstractVector,
    v::AbstractVector,
    jtv::AbstractVector,
)
    copyto!(m.x, x)
    copyto!(m.grad, jtv)
    copyto!(m.con, v)
    NLPModels.jtprod!(m.inner, m.x, m.con, m.grad)
    copyto!(jtv, m.grad)
    return
end

function NLPModels.jac_coord!(
    m::SparseWrapperModel,
    x::AbstractVector,
    jac::AbstractVector,
)
    copyto!(m.x, x)
    NLPModels.jac_coord!(m.inner, m.x, m.jac)
    copyto!(jac, m.jac)
    return
end

function NLPModels.hess_coord!(
    m::M,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x))
) where {M <: SparseWrapperModel}
    copyto!(m.x, x)
    copyto!(m.y, y)
    NLPModels.hess_coord!(m.inner, m.x, m.y, m.hess; obj_weight=obj_weight)
    copyto!(hess, m.hess)
    return
end

function NLPModels.jac_dense!(
    m::Model,
    x::V,
    jac::M
) where {Model <: DenseWrapperModel, V <: AbstractVector, M <: AbstractMatrix}
    copyto!(m.x, x)
    NLPModels.jac_dense!(m.inner, m.x, m.jac)
    copyto!(jac, m.jac)
    return
end

function NLPModels.hess_dense!(
    m::Model,
    x::V,
    y::V,
    hess::M;
    obj_weight = one(eltype(x))
) where {Model <: DenseWrapperModel, V <: AbstractVector, M <: AbstractMatrix}
    copyto!(m.x, x)
    copyto!(m.y, y)
    NLPModels.hess_dense!(m.inner, m.x, m.y, m.hess; obj_weight=obj_weight)
    copyto!(hess, m.hess)
    return
end
