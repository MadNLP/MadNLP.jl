# PlasmoNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

mutable struct NonlinearProgram
    n::Int 
    m::Int
    nnz_hess::Int
    nnz_jac::Int

    obj_val::Float64
    x::StrideOneVector{Float64}
    g::StrideOneVector{Float64}
    l::StrideOneVector{Float64} 
    zl::StrideOneVector{Float64} 
    zu::StrideOneVector{Float64} 
    
    xl::StrideOneVector{Float64} 
    xu::StrideOneVector{Float64} 
    gl::StrideOneVector{Float64} 
    gu::StrideOneVector{Float64}
    
    obj::Function
    obj_grad!::Function
    con!::Function
    con_jac!::Function
    lag_hess!::Function
    
    hess_sparsity!::Function
    jac_sparsity!::Function

    status::Status
    ext::Dict{Symbol,Any}
end

function string(nlp::NonlinearProgram)
    """
    Nonlinear program

    number of variables......................: $(nlp.n)
    number of constraints....................: $(nlp.m)
    number of nonzeros in lagrangian hessian.: $(nlp.nnz_hess)
    number of nonzeros in constraint jacobian: $(nlp.nnz_jac)
    status...................................: $(nlp.status)
    """
end
print(io::IO,nlp::NonlinearProgram) = print(io, string(nlp))
show(io::IO,::MIME"text/plain",nlp::NonlinearProgram) = print(io,nlp)
