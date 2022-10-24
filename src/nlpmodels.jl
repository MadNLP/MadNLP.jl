
function _set_scaling!(con_scale::AbstractVector, jac::SparseMatrixCOO)
    @simd for i in 1:nnz(jac)
        row = @inbounds jac.I[i]
        @inbounds con_scale[row] = max(con_scale[row], abs(jac.V[i]))
    end
end
function _set_scaling!(con_scale::AbstractVector, jac::Matrix)
    for row in 1:size(jac, 1)
        for col in 1:size(jac, 2)
            @inbounds con_scale[row] = max(con_scale[row], abs(jac[row, col]))
        end
    end
end

"""
    scale_constraints!(
        nlp::AbstractNLPModel,
        con_scale::AbstractVector,
        jac::AbstractMatrix;
        max_gradient=1e-8,
    )

Compute the scaling of the constraints associated
to the nonlinear model `nlp`. By default, Ipopt's scaling
is applied. The user can write its own function to scale
appropriately any custom `AbstractNLPModel`.

### Notes

This function assumes that the Jacobian `jac` has been evaluated
before calling this function.

"""
function scale_constraints!(
    nlp::AbstractNLPModel{T},
    con_scale::AbstractVector,
    jac::AbstractMatrix;
    max_gradient=1e-8,
) where T
    fill!(con_scale, zero(T))
    _set_scaling!(con_scale, jac)
    @inbounds for i in eachindex(con_scale)
        con_scale[i] = min(one(T), max_gradient / con_scale[i])
    end
end

"""
    scale_objective(
        nlp::AbstractNLPModel,
        grad::AbstractVector;
        max_gradient=1e-8,
    )

Compute the scaling of the objective associated to the
nonlinear model `nlp`. By default, Ipopt's scaling
is applied. The user can write its own function to scale
appropriately the objective of any custom `AbstractNLPModel`.

### Notes

This function assumes that the gradient `gradient` has been evaluated
before calling this function.

"""
function scale_objective(
    nlp::AbstractNLPModel{T},
    grad::AbstractVector;
    max_gradient=1e-8,
) where T
    return min(one(T), max_gradient / normInf(grad))
end

function get_index_constraints(nlp::AbstractNLPModel; fixed_variable_treatment=MAKE_PARAMETER)
    ind_ineq = findall(get_lcon(nlp) .!= get_ucon(nlp))
    xl = [get_lvar(nlp);view(get_lcon(nlp),ind_ineq)]
    xu = [get_uvar(nlp);view(get_ucon(nlp),ind_ineq)]
    if fixed_variable_treatment == MAKE_PARAMETER
        ind_fixed = findall(xl .== xu)
        ind_lb = findall((xl .!= -Inf) .* (xl .!= xu))
        ind_ub = findall((xu .!=  Inf) .* (xl .!= xu))
    else
        ind_fixed = Int[]
        ind_lb = findall(xl .!=-Inf)
        ind_ub = findall(xu .!= Inf)
    end

    ind_llb = findall((get_lvar(nlp) .!= -Inf).*(get_uvar(nlp) .== Inf))
    ind_uub = findall((get_lvar(nlp) .== -Inf).*(get_uvar(nlp) .!= Inf))

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

