# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module Mc19

import ..MadNLP:
    AbstractLinearSystemScaler, SparseMatrixCOO, SparseMatrixCSC, SubVector, 
    get_tril_to_full, findIJ, libhsl, rescale!

mc19ad(n::Int32,nz::Int32,V::Vector{Cdouble},I::Array{Int32,1},J::Array{Int32,1},
       r::Array{Cfloat,1},c::Array{Cfloat,1},w::Array{Cfloat,1})=ccall(
           (:mc19ad_,libhsl),
           Nothing,
           (Ref{Int32},Ref{Int32},Ptr{Cdouble},Ptr{Int32},Ptr{Int32},
            Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),
           n,nz,V,I,J,r,c,w)
mutable struct Scaler <: AbstractLinearSystemScaler
    tril::SparseMatrixCSC{Float64,Int32}
    full::SparseMatrixCOO{Float64,Int32}
    tril_to_full_view::SubVector{Float64}

    s::Vector{Float64}
    r::Array{Float32,1}
    c::Array{Float32,1}
    w::Array{Float32,1}
end
function Scaler(csc::SparseMatrixCSC{Float64,Int32})
    full,tril_to_full_view = get_tril_to_full(csc)
    I,J = findIJ(full)
    full=SparseMatrixCOO{Float64,Int32}(csc.m,csc.n,I,J,full.nzval)

    s=Vector{Float64}(undef,csc.n)
    r=Array{Float32,1}(undef,csc.n)
    c=Array{Float32,1}(undef,csc.n)
    w=Array{Float32,1}(undef,csc.n*5)
    
    return Scaler(csc,full,tril_to_full_view,s,r,c,w)
end
function rescale!(S::Scaler)
    S.full.V.=S.tril_to_full_view
    mc19ad(Int32(S.full.n),Int32(length(S.full.I)),S.full.V,S.full.I,S.full.J,S.r,S.c,S.w)
    S.s .= exp.((S.r.+S.c)./2)
    (sum(S.s) == Inf || maximum(S.s) > 1e40) && (S.s.=1.)
    return S
end
end # module

# forgiving names
mc19=Mc19
MC19=Mc19
