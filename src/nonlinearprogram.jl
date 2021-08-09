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

# Utils
function get_index_constraints(nlp::NonlinearProgram; fixed_variable_treatment=MAKE_PARAMETER)
    ind_ineq = findall(nlp.gl .!= nlp.gu)
    xl = [nlp.xl;view(nlp.gl,ind_ineq)]
    xu = [nlp.xu;view(nlp.gu,ind_ineq)]
    if fixed_variable_treatment == MAKE_PARAMETER
        ind_fixed = findall(xl .== xu)
        ind_lb = findall((xl .!= -Inf) .* (xl .!= xu))
        ind_ub = findall((xu .!=  Inf) .* (xl .!= xu))
    else
        ind_fixed = Int[]
        ind_lb = findall(xl .!=-Inf)
        ind_ub = findall(xu .!= Inf)
    end

    ind_llb = findall((nlp.xl .== -Inf).*(nlp.xu .!= Inf))
    ind_uub = findall((nlp.xl .!= -Inf).*(nlp.xu .== Inf))

    # Return named tuple
    return (
        ind_ineq=ind_ineq,
        ind_fixed=ind_fixed,
        ind_lb=ind_lb,
        ind_ub=ind_ub,
        ind_llb=ind_llb,
        ind_uub=ind_uub,
    )
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
