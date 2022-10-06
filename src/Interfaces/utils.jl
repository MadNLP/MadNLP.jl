# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# !!! warning
#
#     The contents of this file are experimental.
#
#     Until this message is removed, breaking changes to the functions and types,
#     including their deletion, may be introduced in any minor or patch release of Ipopt.

@enum(
    _FunctionType,
    _kFunctionTypeVariableIndex,
    _kFunctionTypeScalarAffine,
    _kFunctionTypeScalarQuadratic,
)

@enum(
    _BoundType,
    _kBoundTypeLessThan,
    _kBoundTypeGreaterThan,
    _kBoundTypeEqualTo,
)

mutable struct QPBlockData{T}
    objective_type::_FunctionType
    objective_constant::T
    objective_linear_columns::Vector{Int}
    objective_linear_coefficients::Vector{T}
    objective_hessian_structure::Vector{Tuple{Int,Int}}
    objective_hessian_coefficients::Vector{T}

    linear_row_ends::Vector{Int}
    linear_jacobian_structure::Vector{Tuple{Int,Int}}
    linear_coefficients::Vector{T}

    quadratic_row_ends::Vector{Int}
    hessian_structure::Vector{Tuple{Int,Int}}
    quadratic_coefficients::Vector{T}

    g_L::Vector{T}
    g_U::Vector{T}
    mult_g::Vector{Union{Nothing,T}}
    function_type::Vector{_FunctionType}
    bound_type::Vector{_BoundType}

    function QPBlockData{T}() where {T}
        return new(
            # Objective coefficients
            _kFunctionTypeScalarAffine,
            zero(T),
            Int[],
            T[],
            Tuple{Int,Int}[],
            T[],
            # Linear constraints
            Int[],
            Tuple{Int,Int}[],
            T[],
            # Affine constraints
            Int[],
            Tuple{Int,Int}[],
            T[],
            # Bounds
            T[],
            T[],
            Union{Nothing,T}[],
            _FunctionType[],
            _BoundType[],
        )
    end
end

Base.length(block::QPBlockData) = length(block.bound_type)

function _set_objective(block::QPBlockData{T}, f::MOI.VariableIndex) where {T}
    push!(block.objective_linear_columns, f.value)
    push!(block.objective_linear_coefficients, one(T))
    return zero(T)
end

function _set_objective(
    block::QPBlockData{T},
    f::MOI.ScalarAffineFunction{T},
) where {T}
    _set_objective(block, f.terms)
    return f.constant
end

function _set_objective(
    block::QPBlockData{T},
    f::MOI.ScalarQuadraticFunction{T},
) where {T}
    _set_objective(block, f.affine_terms)
    for term in f.quadratic_terms
        i, j = term.variable_1.value, term.variable_2.value
        push!(block.objective_hessian_structure, (i, j))
        push!(block.objective_hessian_coefficients, term.coefficient)
    end
    return f.constant
end

function _set_objective(
    block::QPBlockData{T},
    terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    for term in terms
        push!(block.objective_linear_columns, term.variable.value)
        push!(block.objective_linear_coefficients, term.coefficient)
    end
    return
end

function MOI.set(
    block::QPBlockData{T},
    ::MOI.ObjectiveFunction{F},
    func::F,
) where {
    T,
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{T},
        MOI.ScalarQuadraticFunction{T},
    },
}
    empty!(block.objective_hessian_structure)
    empty!(block.objective_hessian_coefficients)
    empty!(block.objective_linear_columns)
    empty!(block.objective_linear_coefficients)
    block.objective_constant = _set_objective(block, func)
    block.objective_type = _function_info(func)
    return
end

function MOI.get(block::QPBlockData{T}, ::MOI.ObjectiveFunctionType) where {T}
    return _function_type_to_set(T, block.objective_type)
end

function MOI.get(block::QPBlockData{T}, ::MOI.ObjectiveFunction{F}) where {T,F}
    affine_terms = MOI.ScalarAffineTerm{T}[
        MOI.ScalarAffineTerm(
            block.objective_linear_coefficients[i],
            MOI.VariableIndex(x),
        ) for (i, x) in enumerate(block.objective_linear_columns)
    ]
    quadratic_terms = MOI.ScalarQuadraticTerm{T}[]
    for (i, coef) in enumerate(block.objective_hessian_coefficients)
        r, c = block.objective_hessian_structure[i]
        push!(
            quadratic_terms,
            MOI.ScalarQuadraticTerm(
                coef,
                MOI.VariableIndex(r),
                MOI.VariableIndex(c),
            ),
        )
    end
    obj = MOI.ScalarQuadraticFunction(
        quadratic_terms,
        affine_terms,
        block.objective_constant,
    )
    return convert(F, obj)
end

function MOI.get(
    block::QPBlockData{T},
    ::MOI.ListOfConstraintTypesPresent,
) where {T}
    constraints = Set{Tuple{Type,Type}}()
    for i in 1:length(block)
        F = _function_type_to_set(T, block.function_type[i])
        S = _bound_type_to_set(T, block.bound_type[i])
        push!(constraints, (F, S))
    end
    return collect(constraints)
end

function MOI.is_valid(
    block::QPBlockData{T},
    ci::MOI.ConstraintIndex{F,S},
) where {
    T,
    F<:Union{MOI.ScalarAffineFunction{T},MOI.ScalarQuadraticFunction{T}},
    S<:Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T}},
}
    return 1 <= ci.value <= length(block)
end

function MOI.get(
    block::QPBlockData{T},
    ::MOI.ListOfConstraintIndices{F,S},
) where {
    T,
    F<:Union{MOI.ScalarAffineFunction{T},MOI.ScalarQuadraticFunction{T}},
    S<:Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T}},
}
    ret = MOI.ConstraintIndex{F,S}[]
    for i in 1:length(block)
        if _bound_type_to_set(T, block.bound_type[i]) != S
            continue
        elseif _function_type_to_set(T, block.function_type[i]) != F
            continue
        end
        push!(ret, MOI.ConstraintIndex{F,S}(i))
    end
    return ret
end

function MOI.get(
    block::QPBlockData{T},
    ::MOI.NumberOfConstraints{F,S},
) where {
    T,
    F<:Union{MOI.ScalarAffineFunction{T},MOI.ScalarQuadraticFunction{T}},
    S<:Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T}},
}
    return length(MOI.get(block, MOI.ListOfConstraintIndices{F,S}()))
end

function _bound_type_to_set(::Type{T}, k::_BoundType) where {T}
    if k == _kBoundTypeEqualTo
        return MOI.EqualTo{T}
    elseif k == _kBoundTypeLessThan
        return MOI.LessThan{T}
    else
        @assert k == _kBoundTypeGreaterThan
        return MOI.GreaterThan{T}
    end
end

function _function_type_to_set(::Type{T}, k::_FunctionType) where {T}
    if k == _kFunctionTypeVariableIndex
        return MOI.VariableIndex
    elseif k == _kFunctionTypeScalarAffine
        return MOI.ScalarAffineFunction{T}
    else
        @assert k == _kFunctionTypeScalarQuadratic
        return MOI.ScalarQuadraticFunction{T}
    end
end

_function_info(::MOI.VariableIndex) = _kFunctionTypeVariableIndex
_function_info(::MOI.ScalarAffineFunction) = _kFunctionTypeScalarAffine
_function_info(::MOI.ScalarQuadraticFunction) = _kFunctionTypeScalarQuadratic

_set_info(s::MOI.LessThan) = _kBoundTypeLessThan, -Inf, s.upper
_set_info(s::MOI.GreaterThan) = _kBoundTypeGreaterThan, s.lower, Inf
_set_info(s::MOI.EqualTo) = _kBoundTypeEqualTo, s.value, s.value

function _add_function(
    block::QPBlockData{T},
    f::MOI.ScalarAffineFunction{T},
) where {T}
    _add_function(block, f.terms)
    push!(block.quadratic_row_ends, length(block.quadratic_coefficients))
    return _kFunctionTypeScalarAffine, f.constant
end

function _add_function(
    block::QPBlockData{T},
    f::MOI.ScalarQuadraticFunction{T},
) where {T}
    _add_function(block, f.affine_terms)
    for term in f.quadratic_terms
        i, j = term.variable_1.value, term.variable_2.value
        push!(block.hessian_structure, (i, j))
        push!(block.quadratic_coefficients, term.coefficient)
    end
    push!(block.quadratic_row_ends, length(block.quadratic_coefficients))
    return _kFunctionTypeScalarQuadratic, f.constant
end

function _add_function(
    block::QPBlockData{T},
    terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    row = length(block) + 1
    for term in terms
        push!(block.linear_jacobian_structure, (row, term.variable.value))
        push!(block.linear_coefficients, term.coefficient)
    end
    push!(block.linear_row_ends, length(block.linear_jacobian_structure))
    return
end

function MOI.add_constraint(
    block::QPBlockData{T},
    f::Union{MOI.ScalarAffineFunction{T},MOI.ScalarQuadraticFunction{T}},
    set::Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T}},
) where {T}
    function_type, constant = _add_function(block, f)
    bound_type, l, u = _set_info(set)
    push!(block.g_L, l - constant)
    push!(block.g_U, u - constant)
    push!(block.mult_g, nothing)
    push!(block.bound_type, bound_type)
    push!(block.function_type, function_type)
    return MOI.ConstraintIndex{typeof(f),typeof(set)}(length(block.bound_type))
end

function MOI.get(
    block::QPBlockData{T},
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{F,S},
) where {T,F,S}
    row = c.value
    offset = row == 1 ? 1 : (block.linear_row_ends[row-1] + 1)
    affine_terms = MOI.ScalarAffineTerm{T}[
        MOI.ScalarAffineTerm(
            block.linear_coefficients[i],
            MOI.VariableIndex(block.linear_jacobian_structure[i][2]),
        ) for i in offset:block.linear_row_ends[row]
    ]
    quadratic_terms = MOI.ScalarQuadraticTerm{T}[]
    offset = row == 1 ? 1 : (block.quadratic_row_ends[row-1] + 1)
    for i in offset:block.quadratic_row_ends[row]
        r, c = block.hessian_structure[i]
        push!(
            quadratic_terms,
            MOI.ScalarQuadraticTerm(
                block.quadratic_coefficients[i],
                MOI.VariableIndex(r),
                MOI.VariableIndex(c),
            ),
        )
    end
    if length(quadratic_terms) == 0
        return MOI.ScalarAffineFunction(affine_terms, zero(T))
    end
    return MOI.ScalarQuadraticFunction(quadratic_terms, affine_terms, zero(T))
end

function MOI.get(
    block::QPBlockData{T},
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,S},
) where {T,F,S}
    row = c.value
    if block.bound_type[row] == _kBoundTypeEqualTo
        return MOI.EqualTo(block.g_L[row])
    elseif block.bound_type[row] == _kBoundTypeLessThan
        return MOI.LessThan(block.g_U[row])
    else
        @assert block.bound_type[row] == _kBoundTypeGreaterThan
        return MOI.GreaterThan(block.g_L[row])
    end
end

function MOI.set(
    block::QPBlockData{T},
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,MOI.LessThan{T}},
    set::MOI.LessThan{T},
) where {T,F}
    row = c.value
    block.g_U[row] = set.upper
    return
end

function MOI.set(
    block::QPBlockData{T},
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,MOI.GreaterThan{T}},
    set::MOI.GreaterThan{T},
) where {T,F}
    row = c.value
    block.g_L[row] = set.lower
    return
end

function MOI.set(
    block::QPBlockData{T},
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,MOI.EqualTo{T}},
    set::MOI.EqualTo{T},
) where {T,F}
    row = c.value
    block.g_L[row] = set.value
    block.g_U[row] = set.value
    return
end

function MOI.get(
    block::QPBlockData{T},
    ::MOI.ConstraintDualStart,
    c::MOI.ConstraintIndex{F,S},
) where {T,F,S}
    return block.mult_g[c.value]
end

function MOI.set(
    block::QPBlockData{T},
    ::MOI.ConstraintDualStart,
    c::MOI.ConstraintIndex{F,S},
    value,
) where {T,F,S}
    block.mult_g[c.value] = value
    return
end

function MOI.eval_objective(
    block::QPBlockData{T},
    x::AbstractVector{T},
) where {T}
    y = block.objective_constant
    for (i, c) in enumerate(block.objective_linear_columns)
        y += block.objective_linear_coefficients[i] * x[c]
    end
    for (i, (r, c)) in enumerate(block.objective_hessian_structure)
        if r == c
            y += block.objective_hessian_coefficients[i] * x[r] * x[c] / 2
        else
            y += block.objective_hessian_coefficients[i] * x[r] * x[c]
        end
    end
    return y
end

function MOI.eval_objective_gradient(
    block::QPBlockData{T},
    g::AbstractVector{T},
    x::AbstractVector{T},
) where {T}
    g .= zero(T)
    for (i, c) in enumerate(block.objective_linear_columns)
        g[c] += block.objective_linear_coefficients[i]
    end
    for (i, (r, c)) in enumerate(block.objective_hessian_structure)
        g[r] += block.objective_hessian_coefficients[i] * x[c]
        if r != c
            g[c] += block.objective_hessian_coefficients[i] * x[r]
        end
    end
    return
end

function MOI.eval_constraint(
    block::QPBlockData{T},
    g::AbstractVector{T},
    x::AbstractVector{T},
) where {T}
    for i in 1:length(g)
        g[i] = zero(T)
    end
    for (i, (r, c)) in enumerate(block.linear_jacobian_structure)
        g[r] += block.linear_coefficients[i] * x[c]
    end
    i = 0
    for row in 1:length(block.quadratic_row_ends)
        while i < block.quadratic_row_ends[row]
            i += 1
            r, c = block.hessian_structure[i]
            if r == c
                g[row] += block.quadratic_coefficients[i] * x[r] * x[c] / 2
            else
                g[row] += block.quadratic_coefficients[i] * x[r] * x[c]
            end
        end
    end
    return
end

function MOI.jacobian_structure(block::QPBlockData)
    J = copy(block.linear_jacobian_structure)
    i = 0
    for row in 1:length(block.quadratic_row_ends)
        while i < block.quadratic_row_ends[row]
            i += 1
            r, c = block.hessian_structure[i]
            push!(J, (row, r))
            if r != c
                push!(J, (row, c))
            end
        end
    end
    return J
end

function MOI.eval_constraint_jacobian(
    block::QPBlockData{T},
    J::AbstractVector{T},
    x::AbstractVector{T},
) where {T}
    nterms = 0
    for coef in block.linear_coefficients
        nterms += 1
        J[nterms] = coef
    end
    i = 0
    for row in 1:length(block.quadratic_row_ends)
        while i < block.quadratic_row_ends[row]
            i += 1
            r, c = block.hessian_structure[i]
            nterms += 1
            J[nterms] = block.quadratic_coefficients[i] * x[c]
            if r != c
                nterms += 1
                J[nterms] = block.quadratic_coefficients[i] * x[r]
            end
        end
    end
    return nterms
end

function MOI.hessian_lagrangian_structure(block::QPBlockData)
    return vcat(block.objective_hessian_structure, block.hessian_structure)
end

function MOI.eval_hessian_lagrangian(
    block::QPBlockData{T},
    H::AbstractVector{T},
    ::AbstractVector{T},
    σ::T,
    μ::AbstractVector{T},
) where {T}
    nterms = 0
    for c in block.objective_hessian_coefficients
        nterms += 1
        H[nterms] = σ * c
    end
    i = 0
    for row in 1:length(block.quadratic_row_ends)
        while i < block.quadratic_row_ends[row]
            i += 1
            nterms += 1
            H[nterms] = μ[row] * block.quadratic_coefficients[i]
        end
    end
    return nterms
end
