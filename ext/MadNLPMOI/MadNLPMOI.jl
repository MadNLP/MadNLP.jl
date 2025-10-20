module MadNLPMOI

import MadNLP, MathOptInterface, NLPModels, PrecompileTools

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

include("utils.jl")

const _PARAMETER_OFFSET = 0x00f0000000000000

_is_parameter(x::MOI.VariableIndex) = x.value >= _PARAMETER_OFFSET
_is_parameter(term::MOI.ScalarAffineTerm) = _is_parameter(term.variable)
_is_parameter(term::MOI.ScalarQuadraticTerm) = _is_parameter(term.variable_1) || _is_parameter(term.variable_2)

"""
    Optimizer()

Create a new MadNLP optimizer.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    solver::Union{Nothing,MadNLP.MadNLPSolver}
    nlp::Union{Nothing,NLPModels.AbstractNLPModel}
    result::Union{Nothing,MadNLP.MadNLPExecutionStats{Float64}}

    name::String
    invalid_model::Bool
    silent::Bool
    options::Dict{Symbol,Any}
    solve_time::Float64
    solve_iterations::Int
    sense::MOI.OptimizationSense

    parameters::Dict{MOI.VariableIndex,MOI.Nonlinear.ParameterIndex}
    variables::MOI.Utilities.VariablesContainer{Float64}
    list_of_variable_indices::Vector{MOI.VariableIndex}
    variable_primal_start::Vector{Union{Nothing,Float64}}
    mult_x_L::Vector{Union{Nothing,Float64}}
    mult_x_U::Vector{Union{Nothing,Float64}}

    nlp_data::MOI.NLPBlockData
    nlp_dual_start::Union{Nothing,Vector{Float64}}

    qp_data::QPBlockData{Float64}
    nlp_model::Union{Nothing,MOI.Nonlinear.Model}
end

function MadNLP.Optimizer(; kwargs...)
    option_dict = Dict{Symbol, Any}()
    for (name, value) in kwargs
        option_dict[name] = value
    end
    return Optimizer(
        nothing,
        nothing,
        nothing,
        "",
        false,
        false,
        option_dict,
        NaN,
        0,
        MOI.FEASIBILITY_SENSE,
        Dict{MOI.VariableIndex,Float64}(),
        MOI.Utilities.VariablesContainer{Float64}(),
        MOI.VariableIndex[],
        Union{Nothing,Float64}[],
        Union{Nothing,Float64}[],
        Union{Nothing,Float64}[],
        MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
        nothing,
        QPBlockData{Float64}(),
        nothing,
    )
end

const _SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

const _FUNCTIONS = Union{
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
}

MOI.get(::Optimizer, ::MOI.SolverVersion) = MadNLP.version()

### _EmptyNLPEvaluator

struct _EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.features_available(::_EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::_EmptyNLPEvaluator, ::Any) = nothing
MOI.eval_constraint(::_EmptyNLPEvaluator, g, x) = nothing
MOI.jacobian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.hessian_lagrangian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.eval_constraint_jacobian(::_EmptyNLPEvaluator, J, x) = nothing
MOI.eval_constraint_jacobian_transpose_product(::_EmptyNLPEvaluator, Jtv, x, v) = nothing
MOI.eval_hessian_lagrangian(::_EmptyNLPEvaluator, H, x, σ, μ) = nothing

function MOI.empty!(model::Optimizer)
    model.solver = nothing
    model.nlp = nothing
    model.result = nothing
    model.invalid_model = false
    model.solve_time = NaN
    model.solve_iterations = 0
    model.sense = MOI.FEASIBILITY_SENSE
    empty!(model.parameters)
    MOI.empty!(model.variables)
    empty!(model.list_of_variable_indices)
    empty!(model.variable_primal_start)
    empty!(model.mult_x_L)
    empty!(model.mult_x_U)
    model.nlp_data = MOI.NLPBlockData([], _EmptyNLPEvaluator(), false)
    model.nlp_dual_start = nothing
    model.qp_data = QPBlockData{Float64}()
    model.nlp_model = nothing
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.variables) &&
           isempty(model.variable_primal_start) &&
           isempty(model.mult_x_L) &&
           isempty(model.mult_x_U) &&
           model.nlp_data.evaluator isa _EmptyNLPEvaluator &&
           model.sense == MOI.FEASIBILITY_SENSE
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

function _init_nlp_model(model)
    if model.nlp_model === nothing
        if !(model.nlp_data.evaluator isa _EmptyNLPEvaluator)
            error("Cannot mix the new and legacy nonlinear APIs")
        end
        model.nlp_model = MOI.Nonlinear.Model()
    end
    return
end

MOI.get(::Optimizer, ::MOI.SolverName) = "MadNLP"

function MOI.supports_add_constrained_variable(
    ::Optimizer,
    ::Type{MOI.Parameter{Float64}},
)
    return true
end

function MOI.add_constrained_variable(
    model::Optimizer,
    set::MOI.Parameter{Float64},
)
    if model.nlp_model === nothing
        model.nlp_model = MOI.Nonlinear.Model()
    end
    p = MOI.VariableIndex(_PARAMETER_OFFSET + length(model.parameters))
    push!(model.list_of_variable_indices, p)
    model.parameters[p] =
        MOI.Nonlinear.add_parameter(model.nlp_model, set.value)
    ci = MOI.ConstraintIndex{MOI.VariableIndex,typeof(set)}(p.value)
    return p, ci
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}},
    set::MOI.Parameter{Float64},
)
    p = model.parameters[MOI.VariableIndex(ci.value)]
    model.nlp_model[p] = set.value
    return
end

_replace_parameters(model::Optimizer, f) = f

function _replace_parameters(model::Optimizer, f::MOI.VariableIndex)
    if _is_parameter(f)
        return model.parameters[f]
    end
    return f
end

function _replace_parameters(model::Optimizer, f::MOI.ScalarAffineFunction)
    if any(_is_parameter, f.terms)
        g = convert(MOI.ScalarNonlinearFunction, f)
        return _replace_parameters(model, g)
    end
    return f
end

function _replace_parameters(model::Optimizer, f::MOI.ScalarQuadraticFunction)
    if any(_is_parameter, f.affine_terms) ||
       any(_is_parameter, f.quadratic_terms)
        g = convert(MOI.ScalarNonlinearFunction, f)
        return _replace_parameters(model, g)
    end
    return f
end

function _replace_parameters(model::Optimizer, f::MOI.ScalarNonlinearFunction)
    for (i, arg) in enumerate(f.args)
        f.args[i] = _replace_parameters(model, arg)
    end
    return f
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.VariableIndex,_FUNCTIONS}},
    ::Type{<:_SETS},
)
    return true
end

### MOI.ListOfConstraintTypesPresent

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintTypesPresent)
    ret = MOI.get(model.variables, attr)
    append!(ret, MOI.get(model.qp_data, attr))
    return ret
end

### MOI.Name

MOI.supports(::Optimizer, ::MOI.Name) = true

function MOI.set(model::Optimizer, ::MOI.Name, value::String)
    model.name = value
    return
end

MOI.get(model::Optimizer, ::MOI.Name) = model.name

### MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

### MOI.TimeLimitSec

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawOptimizerAttribute("max_cpu_time"), Float64(value))
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    delete!(model.options, :max_cpu_time)
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return get(model.options, :max_cpu_time, nothing)
end

### MOI.RawOptimizerAttribute

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    model.options[Symbol(p.name)] = value
    # No need to reset model.solver because this gets handled in optimize!.
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    if !haskey(model.options, p.name)
        error("RawParameter with name $(p.name) is not set.")
    end
    return model.options[p.name]
end

### Variables

"""
    column(x::MOI.VariableIndex)

Return the column associated with a variable.
"""
column(x::MOI.VariableIndex) = x.value

function MOI.add_variable(model::Optimizer)
    push!(model.variable_primal_start, nothing)
    push!(model.mult_x_L, nothing)
    push!(model.mult_x_U, nothing)
    model.solver = nothing
    x = MOI.add_variable(model.variables)
    push!(model.list_of_variable_indices, x)
    return x
end

function MOI.is_valid(model::Optimizer, x::MOI.VariableIndex)
    if _is_parameter(x)
        return haskey(model.parameters, x)
    end
    return MOI.is_valid(model.variables, x)
end

function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{Float64}})
    p = MOI.VariableIndex(ci.value)
    return haskey(model.parameters, p)
end

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return model.list_of_variable_indices
end

function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    return length(model.list_of_variable_indices)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.is_valid(model.variables, ci)
end

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.NumberOfConstraints{MOI.VariableIndex,<:_SETS},
        MOI.ListOfConstraintIndices{MOI.VariableIndex,<:_SETS},
    },
)
    return MOI.get(model.variables, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
    c::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.get(model.variables, attr, c)
end

function MOI.add_constraint(model::Optimizer, x::MOI.VariableIndex, set::_SETS)
    index = MOI.add_constraint(model.variables, x, set)
    model.solver = nothing
    return index
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
    set::S,
) where {S<:_SETS}
    MOI.set(model.variables, MOI.ConstraintSet(), ci, set)
    model.solver = nothing
    return
end

function MOI.delete(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    MOI.delete(model.variables, ci)
    model.solver = nothing
    return
end

### ScalarAffineFunction and ScalarQuadraticFunction constraints

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    return MOI.is_valid(model.qp_data, ci)
end

function MOI.add_constraint(model::Optimizer, func::_FUNCTIONS, set::_SETS)
    index = MOI.add_constraint(model.qp_data, func, set)
    model.solver = nothing
    return index
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.NumberOfConstraints{F,S},MOI.ListOfConstraintIndices{F,S}},
) where {F<:_FUNCTIONS,S<:_SETS}
    return MOI.get(model.qp_data, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.ConstraintFunction,
        MOI.ConstraintSet,
        MOI.ConstraintDualStart,
    },
    c::MOI.ConstraintIndex{F,S},
) where {F<:_FUNCTIONS,S<:_SETS}
    return MOI.get(model.qp_data, attr, c)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{F,S},
    set::S,
) where {F<:_FUNCTIONS,S<:_SETS}
    MOI.set(model.qp_data, MOI.ConstraintSet(), ci, set)
    model.solver = nothing
    return
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F<:_FUNCTIONS,S<:_SETS}
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{F,S},
    value::Union{Real,Nothing},
) where {F<:_FUNCTIONS,S<:_SETS}
    MOI.throw_if_not_valid(model, ci)
    MOI.set(model.qp_data, attr, ci, value)
    # No need to reset model.solver, because this gets handled in optimize!.
    return
end

### ScalarNonlinearFunction

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS},
)
    if model.nlp_model === nothing
        return false
    end
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    return MOI.is_valid(model.nlp_model, index)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarNonlinearFunction,
    s::_SETS,
)
    if model.nlp_model === nothing
        model.nlp_model = MOI.Nonlinear.Model()
    end
    if !isempty(model.parameters)
        _replace_parameters(model, f)
    end
    index = MOI.Nonlinear.add_constraint(model.nlp_model, f, s)
    model.solver = nothing
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(index.value)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
    func::MOI.ScalarNonlinearFunction,
)
    if model.nlp_model === nothing
        model.nlp_model = MOI.Nonlinear.Model()
    end
    if !isempty(model.parameters)
        _replace_parameters(model, func)
    end
    MOI.Nonlinear.set_objective(model.nlp_model, func)
    model.solver = nothing
    return
end

### UserDefinedFunction

MOI.supports(model::Optimizer, ::MOI.UserDefinedFunction) = true

function MOI.set(model::Optimizer, attr::MOI.UserDefinedFunction, args)
    _init_nlp_model(model)
    MOI.Nonlinear.register_operator(
        model.nlp_model,
        attr.name,
        attr.arity,
        args...,
    )
    return
end

### ListOfSupportedNonlinearOperators

function MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators)
    _init_nlp_model(model)
    return MOI.get(model.nlp_model, attr)
end

### MOI.VariablePrimalStart

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    if _is_parameter(vi)
        return  # Do nothing
    end
    MOI.throw_if_not_valid(model, vi)
    model.variable_primal_start[column(vi)] = value
    # No need to reset model.solver, because this gets handled in optimize!.
    return
end

### MOI.ConstraintDualStart

_dual_start(::Optimizer, ::Nothing, ::Int = 1) = 0.0

function _dual_start(model::Optimizer, value::Real, scale::Int = 1)
    return _dual_multiplier(model) * value * scale
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{MOI.ConstraintIndex{MOI.VariableIndex,S}},
) where {S<:_SETS}
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    model.mult_x_L[ci.value] = value
    # No need to reset model.solver, because this gets handled in optimize!.
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return model.mult_x_L[ci.value]
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    model.mult_x_U[ci.value] = value
    # No need to reset model.solver, because this gets handled in optimize!.
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return model.mult_x_U[ci.value]
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    if value === nothing
        model.mult_x_L[ci.value] = nothing
        model.mult_x_U[ci.value] = nothing
    elseif value >= 0.0
        model.mult_x_L[ci.value] = value
        model.mult_x_U[ci.value] = 0.0
    else
        model.mult_x_L[ci.value] = 0.0
        model.mult_x_U[ci.value] = value
    end
    # No need to reset model.solver, because this gets handled in optimize!.
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    l = model.mult_x_L[ci.value]
    u = model.mult_x_U[ci.value]
    return (l === u === nothing) ? nothing : (l + u)
end

### MOI.NLPBlockDualStart

MOI.supports(::Optimizer, ::MOI.NLPBlockDualStart) = true

function MOI.set(
    model::Optimizer,
    ::MOI.NLPBlockDualStart,
    values::Union{Nothing,Vector},
)
    model.nlp_dual_start = values
    # No need to reset model.solver, because this gets handled in optimize!.
    return
end

MOI.get(model::Optimizer, ::MOI.NLPBlockDualStart) = model.nlp_dual_start

### MOI.NLPBlock

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    model.nlp_data = nlp_data
    model.solver = nothing
    return
end

### ObjectiveSense

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    model.sense = sense
    model.solver = nothing
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

### ObjectiveFunction

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ObjectiveFunctionType,MOI.ObjectiveFunction},
)
    return MOI.get(model.qp_data, attr)
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{<:Union{MOI.VariableIndex,<:_FUNCTIONS}},
)
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
    func::F,
) where {F<:Union{MOI.VariableIndex,<:_FUNCTIONS}}
    MOI.set(model.qp_data, attr, func)
    if model.nlp_model !== nothing
        MOI.Nonlinear.set_objective(model.nlp_model, nothing)
    end
    model.solver = nothing
    return
end

### Eval_F_CB

function MOI.eval_objective(model::Optimizer, x)
    if model.sense == MOI.FEASIBILITY_SENSE
        return 0.0
    elseif model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    end
    return MOI.eval_objective(model.qp_data, x)
end

### Eval_Grad_F_CB

function MOI.eval_objective_gradient(model::Optimizer, grad, x)
    if model.sense == MOI.FEASIBILITY_SENSE
        grad .= zero(eltype(grad))
    elseif model.nlp_data.has_objective
        MOI.eval_objective_gradient(model.nlp_data.evaluator, grad, x)
    else
        MOI.eval_objective_gradient(model.qp_data, grad, x)
    end
    return
end

### Eval_G_CB

function MOI.eval_constraint(model::Optimizer, g, x)
    MOI.eval_constraint(model.qp_data, g, x)
    g_nlp = view(g, (length(model.qp_data)+1):length(g))
    MOI.eval_constraint(model.nlp_data.evaluator, g_nlp, x)
    return
end

### Eval_Jac_G_CB

function MOI.jacobian_structure(model::Optimizer)
    J = MOI.jacobian_structure(model.qp_data)
    offset = length(model.qp_data)
    if length(model.nlp_data.constraint_bounds) > 0
        for (row, col) in MOI.jacobian_structure(model.nlp_data.evaluator)
            push!(J, (row + offset, col))
        end
    end
    return J
end

function MOI.eval_constraint_jacobian(model::Optimizer, values, x)
    offset = MOI.eval_constraint_jacobian(model.qp_data, values, x)
    nlp_values = view(values, offset:length(values))
    MOI.eval_constraint_jacobian(model.nlp_data.evaluator, nlp_values, x)
    return
end

function MOI.eval_constraint_jacobian_transpose_product(model::Optimizer, Jtv, x, v)
    MOI.eval_constraint_jacobian_transpose_product(model.nlp_data.evaluator, Jtv, x, v)
    # Evaluate QPBlockData after NLPEvaluator to ensure that Jtv is not reset.
    MOI.eval_constraint_jacobian_transpose_product(model.qp_data, Jtv, x, v)
    return
end

### Eval_H_CB

function MOI.hessian_lagrangian_structure(model::Optimizer)
    H = MOI.hessian_lagrangian_structure(model.qp_data)
    append!(H, MOI.hessian_lagrangian_structure(model.nlp_data.evaluator))
    return H
end

function MOI.eval_hessian_lagrangian(model::Optimizer, H, x, σ, μ)
    offset = MOI.eval_hessian_lagrangian(model.qp_data, H, x, σ, μ)
    H_nlp = view(H, offset:length(H))
    μ_nlp = view(μ, (length(model.qp_data)+1):length(μ))
    MOI.eval_hessian_lagrangian(model.nlp_data.evaluator, H_nlp, x, σ, μ_nlp)
    return
end

### NLPModels wrapper
struct MOIModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    model::Optimizer
    counters::NLPModels.Counters
end

NLPModels.obj(nlp::MOIModel, x::AbstractVector{Float64}) = MOI.eval_objective(nlp.model,x)

function NLPModels.grad!(nlp::MOIModel, x::AbstractVector{Float64}, g::AbstractVector{Float64})
    MOI.eval_objective_gradient(nlp.model, g, x)
end

function NLPModels.cons!(nlp::MOIModel, x::AbstractVector{Float64}, c::AbstractVector{Float64})
    MOI.eval_constraint(nlp.model, c, x)
end

function NLPModels.jac_coord!(nlp::MOIModel, x::AbstractVector{Float64}, jac::AbstractVector{Float64})
    MOI.eval_constraint_jacobian(nlp.model, jac, x)
end

function NLPModels.jtprod!(nlp::MOIModel, x::AbstractVector{Float64}, v::Vector{Float64}, Jtv::AbstractVector{Float64})
    MOI.eval_constraint_jacobian_transpose_product(nlp.model, Jtv, x, v)
end

function NLPModels.hess_coord!(nlp::MOIModel, x::AbstractVector{Float64}, l::AbstractVector{Float64}, hess::AbstractVector{Float64}; obj_weight::Float64=1.0)
    MOI.eval_hessian_lagrangian(nlp.model, hess, x, obj_weight, l)
end

function NLPModels.hess_structure!(nlp::MOIModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    @assert length(I) == length(J) == length(MOI.hessian_lagrangian_structure(nlp.model))
    cnt = 1
    for (row, col) in  MOI.hessian_lagrangian_structure(nlp.model)
        I[cnt], J[cnt] = row, col
        cnt += 1
    end
end

function NLPModels.jac_structure!(nlp::MOIModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    @assert length(I) == length(J) == length(MOI.jacobian_structure(nlp.model))
    cnt = 1
    for (row, col) in  MOI.jacobian_structure(nlp.model)
        I[cnt], J[cnt] = row, col
        cnt += 1
    end
end

### TODO
function MOIModel(model::Optimizer)
    # Check model is nonempty.
    vars = MOI.get(model.variables, MOI.ListOfVariableIndices())
    if isempty(vars)
        model.invalid_model = true
        return
    end
    # Create NLP backend.
    if model.nlp_model !== nothing
        backend = MOI.Nonlinear.SparseReverseMode()
        evaluator = MOI.Nonlinear.Evaluator(model.nlp_model, backend, vars)
        model.nlp_data = MOI.NLPBlockData(evaluator)
    end
    # Check model's structure.
    has_quadratic_constraints =
        any(isequal(_kFunctionTypeScalarQuadratic), model.qp_data.function_type)
    has_nlp_constraints = !isempty(model.nlp_data.constraint_bounds)
    has_nlp_objective = model.nlp_data.has_objective
    has_hessian = :Hess in MOI.features_available(model.nlp_data.evaluator)
    is_nlp = has_nlp_constraints || has_nlp_objective
    # Initialize evaluator using model's structure.
    init_feat = [:Grad]
    if has_hessian
        push!(init_feat, :Hess)
    end
    if has_nlp_constraints
        push!(init_feat, :Jac)
    end
    MOI.initialize(model.nlp_data.evaluator, init_feat)

    # Initial variable
    nvar = length(model.variables.lower)
    x0  = zeros(nvar)
    for i in 1:length(model.variable_primal_start)
        x0[i] = if model.variable_primal_start[i] !== nothing
            model.variable_primal_start[i]
        else
            clamp(0.0, model.variables.lower[i], model.variables.upper[i])
        end
    end

    # Constraints bounds
    g_L, g_U = copy(model.qp_data.g_L), copy(model.qp_data.g_U)
    for bound in model.nlp_data.constraint_bounds
        push!(g_L, bound.lower)
        push!(g_U, bound.upper)
    end
    ncon = length(g_L)

    # Sparsity
    jacobian_sparsity = MOI.jacobian_structure(model)
    hessian_sparsity = if has_hessian
        MOI.hessian_lagrangian_structure(model)
    else
        Tuple{Int,Int}[]
    end
    nnzh = length(hessian_sparsity)
    nnzj = length(jacobian_sparsity)

    # Dual multipliers
    y0 = zeros(ncon)
    for (i, start) in enumerate(model.qp_data.mult_g)
        y0[i] = _dual_start(model, start, -1)
    end
    offset = length(model.qp_data.mult_g)
    if model.nlp_dual_start === nothing
        y0[(offset+1):end] .= 0.0
    else
        for (i, start) in enumerate(model.nlp_dual_start::Vector{Float64})
            y0[offset+i] = _dual_start(model, start, -1)
        end
    end
    # TODO: initial bounds' multipliers.

    return MOIModel(
        NLPModels.NLPModelMeta(
            nvar,
            x0 = x0,
            lvar = model.variables.lower,
            uvar = model.variables.upper,
            ncon = ncon,
            y0 = y0,
            lcon = g_L,
            ucon = g_U,
            nnzj = nnzj,
            nnzh = nnzh,
            minimize = model.sense == MOI.MIN_SENSE
        ),
        model,
        NLPModels.Counters(),
    )
end

function copy_parameters(model::Optimizer)
    if model.nlp_model === nothing
        return
    end
    empty!(model.qp_data.parameters)
    for (p, index) in model.parameters
        model.qp_data.parameters[p.value] = model.nlp_model[index]
    end
    return
end

function MOI.optimize!(model::Optimizer)
    model.nlp = MOIModel(model)
    if model.invalid_model
        return
    end
    copy_parameters(model)
    if model.silent
        model.options[:print_level] = MadNLP.ERROR
    end
    options = copy(model.options)
    # Specific options depending on problem's structure.
    # 1/ Switch to LBFGS if the Hessian is not specified.
    has_hessian = :Hess in MOI.features_available(model.nlp_data.evaluator)
    if !has_hessian
        options[:hessian_approximation] = MadNLP.CompactLBFGS
    end
    # 2/ Set Jacobian to constant if all constraints are linear.
    has_nlp_constraints =
        any(isequal(_kFunctionTypeScalarQuadratic), model.qp_data.function_type) ||
        !isempty(model.nlp_data.constraint_bounds)
    if !has_nlp_constraints
        options[:jacobian_constant] = true
    end
    # Instantiate MadNLP.
    model.solver = MadNLP.MadNLPSolver(model.nlp; options...)
    model.result = MadNLP.solve!(model.solver)
    model.solve_time = model.solver.cnt.total_time
    model.solve_iterations = model.solver.cnt.k
    return
end

const _STATUS_CODES = Dict{MadNLP.Status,MOI.TerminationStatusCode}(
    MadNLP.SOLVE_SUCCEEDED => MOI.LOCALLY_SOLVED,
    MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL => MOI.ALMOST_LOCALLY_SOLVED,
    MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL => MOI.SLOW_PROGRESS,
    MadNLP.DIVERGING_ITERATES => MOI.INFEASIBLE_OR_UNBOUNDED,
    MadNLP.INFEASIBLE_PROBLEM_DETECTED => MOI.LOCALLY_INFEASIBLE,
    MadNLP.MAXIMUM_ITERATIONS_EXCEEDED => MOI.ITERATION_LIMIT,
    MadNLP.MAXIMUM_WALLTIME_EXCEEDED => MOI.TIME_LIMIT,
    MadNLP.INITIAL => MOI.OPTIMIZE_NOT_CALLED,
    MadNLP.RESTORATION_FAILED => MOI.NUMERICAL_ERROR,
    MadNLP.INVALID_NUMBER_DETECTED => MOI.INVALID_MODEL,
    MadNLP.ERROR_IN_STEP_COMPUTATION => MOI.NUMERICAL_ERROR,
    MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM => MOI.INVALID_MODEL,
    MadNLP.USER_REQUESTED_STOP => MOI.INTERRUPTED,
    MadNLP.INTERNAL_ERROR => MOI.OTHER_ERROR,
    MadNLP.INVALID_NUMBER_OBJECTIVE => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_GRADIENT => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_CONSTRAINTS => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_JACOBIAN => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN => MOI.INVALID_MODEL,
)

### MOI.ResultCount

# Ipopt always has an iterate available.
function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return (model.solver !== nothing) ? 1 : 0
end

### MOI.TerminationStatus

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.invalid_model
        return MOI.INVALID_MODEL
    elseif model.solver === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return get(_STATUS_CODES, model.result.status, MOI.OTHER_ERROR)
end

### MOI.RawStatusString

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.invalid_model
        return "The model has no variable"
    elseif model.solver === nothing
        return "Optimize not called"
    end
    return MadNLP.get_status_output(model.result.status, model.result.options)
end


### MOI.PrimalStatus

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = model.result.status
    if status == MadNLP.SOLVE_SUCCEEDED
        return MOI.FEASIBLE_POINT
    elseif status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == MadNLP.INFEASIBLE_PROBLEM_DETECTED
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

### MOI.DualStatus

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = model.result.status
    if status == MadNLP.SOLVE_SUCCEEDED
        return MOI.FEASIBLE_POINT
    elseif status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == MadNLP.INFEASIBLE_PROBLEM_DETECTED
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

### MOI.SolveTimeSec

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

### MOI.ObjectiveValue

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    scale = (model.sense == MOI.MAX_SENSE) ? -1 : 1
    return scale * model.result.objective
end

### MOI.VariablePrimal

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    if _is_parameter(vi)
        p = model.parameters[vi]
        return model.nlp_model[p]
    end
    return model.result.solution[vi.value]
end

### MOI.ConstraintPrimal

row(model::Optimizer, ci::MOI.ConstraintIndex{<:_FUNCTIONS}) = ci.value

function row(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction},
)
    return length(model.qp_data) + ci.value
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.result.constraints[row(model, ci)]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.result.solution[ci.value]
end

### MOI.ConstraintDual
_dual_multiplier(model::Optimizer) = 1.0

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    s = -1.0
    return s * model.result.multipliers[ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.result.multipliers_L[ci.value] - model.result.multipliers_U[ci.value]
    return min(0.0, rc * _dual_multiplier(model))
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.result.multipliers_L[ci.value] - model.result.multipliers_U[ci.value]
    return max(0.0, rc * _dual_multiplier(model))
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.result.multipliers_L[ci.value] - model.result.multipliers_U[ci.value]
    return rc
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Interval{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.result.multipliers_L[ci.value] - model.result.multipliers_U[ci.value]
    return rc
end

### MOI.NLPBlockDual

function MOI.get(model::Optimizer, attr::MOI.NLPBlockDual)
    MOI.check_result_index_bounds(model, attr)
    s = -_dual_multiplier(model)
    offset = length(model.qp_data)
    return s .* model.result.multipliers[(offset+1):end]
end

### MOI.BarrierIterations
MOI.get(model::Optimizer,::MOI.BarrierIterations) = model.solve_iterations

MadNLP.@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    MadNLP.@compile_workload begin
        nlp = MadNLP.HS15Model()
        MadNLP.madnlp(nlp; print_level=MadNLP.ERROR)
        precompile(Tuple{typeof(MadNLP.madnlp), NLPModels.AbstractNLPModel{T, S} where S where T})
    end 
end

end # module

