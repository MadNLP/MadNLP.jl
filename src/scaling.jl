
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
    nlp::AbstractNLPModel,
    con_scale::AbstractVector,
    jac::AbstractMatrix;
    max_gradient=1e-8,
)
    fill!(con_scale, 0.0)
    _set_scaling!(con_scale, jac)
    con_scale .= min.(1.0, max_gradient ./ con_scale)
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
    nlp::AbstractNLPModel,
    grad::AbstractVector;
    max_gradient=1e-8,
)
    return min(1, max_gradient / norm(grad, Inf))
end

