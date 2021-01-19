# MadNLP.jl
# Modified from Ipopt.jl (https://github.com/jump-dev/Ipopt.jl)

@kwdef mutable struct VariableInfo
    lower_bound::Float64 = -Inf
    has_lower_bound::Bool = false
    lower_bound_dual_start::Union{Nothing, Float64} = nothing
    upper_bound::Float64 = Inf
    has_upper_bound::Bool = false
    upper_bound_dual_start::Union{Nothing, Float64} = nothing
    is_fixed::Bool = false
    start::Union{Nothing,Float64} = nothing
end

mutable struct ConstraintInfo{F, S}
    func::F
    set::S
    dual_start::Union{Nothing, Float64}
end
ConstraintInfo(func, set) = ConstraintInfo(func, set, nothing)

mutable struct Optimizer <: MOI.AbstractOptimizer
    nlp::Union{NonlinearProgram,Nothing}
    ips::Union{Solver,Nothing}

    # Problem data.
    variable_info::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{
        MOI.SingleVariable,MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64},Nothing}

    linear_le_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}}
    linear_ge_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}}
    linear_eq_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}}
    quadratic_le_constraints::Vector{ConstraintInfo{MOI.ScalarQuadraticFunction{Float64},MOI.LessThan{Float64}}}
    quadratic_ge_constraints::Vector{ConstraintInfo{MOI.ScalarQuadraticFunction{Float64},MOI.GreaterThan{Float64}}}
    quadratic_eq_constraints::Vector{ConstraintInfo{MOI.ScalarQuadraticFunction{Float64},MOI.EqualTo{Float64}}}
    nlp_dual_start::Union{Nothing,Vector{Float64}}

    # Parameters.
    option_dict::Dict{Symbol, Any}

    # Solution attributes.
    solve_time::Float64
end

function Optimizer(;kwargs...)
    option_dict = Dict{Symbol, Any}()
    for (name, value) in kwargs
        option_dict[name] = value
    end
    return Optimizer(nothing,nothing,[],empty_nlp_data(),MOI.FEASIBILITY_SENSE,
                     nothing,[],[],[],[],[],[],nothing,option_dict,NaN)
end

# define empty nlp evaluator
struct EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end
MOI.features_available(::EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::EmptyNLPEvaluator, features) = nothing
MOI.eval_objective(::EmptyNLPEvaluator, x) = NaN
MOI.eval_constraint(::EmptyNLPEvaluator, g, x) = @assert length(g) == 0
MOI.eval_objective_gradient(::EmptyNLPEvaluator, g, x) = fill!(g, 0.0)
MOI.jacobian_structure(::EmptyNLPEvaluator) = Tuple{Int,Int}[]
MOI.hessian_lagrangian_structure(::EmptyNLPEvaluator) = Tuple{Int,Int}[]
MOI.eval_constraint_jacobian(::EmptyNLPEvaluator, J, x) = nothing
MOI.eval_hessian_lagrangian(::EmptyNLPEvaluator, H, x, σ, μ) = nothing
empty_nlp_data() = MOI.NLPBlockData([], EmptyNLPEvaluator(), false)

# for throw_if_valid
MOI.is_valid(model::Optimizer, vi::MOI.VariableIndex) = vi.value in eachindex(model.variable_info)
function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    return MOI.is_valid(model, vi) && has_upper_bound(model, vi)
end
function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    return MOI.is_valid(model, vi) && has_lower_bound(model, vi)
end
function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}})
    vi = MOI.VariableIndex(ci.value)
    return MOI.is_valid(model, vi) && is_fixed(model, vi)
end


# supported obj/var/cons
MOI.supports(::Optimizer, ::MOI.NLPBlock) = true
MOI.supports(::Optimizer,::MOI.ObjectiveFunction{MOI.SingleVariable}) = true
MOI.supports(::Optimizer,::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::Optimizer,::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}) = true
MOI.supports(::Optimizer, ::MOI.VariablePrimalStart,::Type{MOI.VariableIndex}) = true
MOI.supports_constraint(::Optimizer,::Type{MOI.SingleVariable},::Type{MOI.LessThan{Float64}})=true
MOI.supports_constraint(::Optimizer,::Type{MOI.SingleVariable},::Type{MOI.GreaterThan{Float64}})=true
MOI.supports_constraint(::Optimizer,::Type{MOI.SingleVariable},::Type{MOI.EqualTo{Float64}})=true
MOI.supports_constraint(::Optimizer,::Type{MOI.ScalarAffineFunction{Float64}},::Type{MOI.LessThan{Float64}})=true
MOI.supports_constraint(::Optimizer,::Type{MOI.ScalarAffineFunction{Float64}},::Type{MOI.GreaterThan{Float64}})=true
MOI.supports_constraint(::Optimizer,::Type{MOI.ScalarAffineFunction{Float64}},::Type{MOI.EqualTo{Float64}})=true
MOI.supports_constraint(::Optimizer,::Type{MOI.ScalarQuadraticFunction{Float64}},::Type{MOI.LessThan{Float64}})=true
MOI.supports_constraint(::Optimizer,::Type{MOI.ScalarQuadraticFunction{Float64}},::Type{MOI.GreaterThan{Float64}})=true
MOI.supports_constraint(::Optimizer,::Type{MOI.ScalarQuadraticFunction{Float64}},::Type{MOI.EqualTo{Float64}})=true

## TODO
MOI.supports_constraint(::Optimizer,::Type{MOI.ScalarAffineFunction{Float64}},::Type{MOI.Interval{Float64}})=false
MOI.supports_constraint(::Optimizer,::Type{MOI.ScalarQuadraticFunction{Float64}},::Type{MOI.Interval{Float64}})=false

MOI.get(::Optimizer, ::MOI.SolverName) = "MadNLP"
MOI.get(model::Optimizer,::MOI.ObjectiveFunctionType)=typeof(model.objective)
MOI.get(model::Optimizer,::MOI.NumberOfVariables)=length(model.variable_info)
MOI.get(model::Optimizer,::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}})=length(model.linear_le_constraints)
MOI.get(model::Optimizer,::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}})=length(model.linear_eq_constraints)
MOI.get(model::Optimizer,::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}})=length(model.linear_ge_constraints)
MOI.get(model::Optimizer,::MOI.NumberOfConstraints{MOI.SingleVariable,MOI.LessThan{Float64}})=count(e->e.has_upper_bound,model.variable_info)
MOI.get(model::Optimizer,::MOI.NumberOfConstraints{MOI.SingleVariable,MOI.EqualTo{Float64}})=count(e->e.is_fixed,model.variable_info)
MOI.get(model::Optimizer,::MOI.NumberOfConstraints{MOI.SingleVariable,MOI.GreaterThan{Float64}})=count(e->e.has_lower_bound,model.variable_info)
MOI.get(model::Optimizer,::MOI.ObjectiveFunction) = model.objective
MOI.get(model::Optimizer,::MOI.ListOfVariableIndices) = [MOI.VariableIndex(i) for i in 1:length(model.variable_info)]

macro define_get_con_attr(attr,function_type, set_type, prefix, attrfield)
    array_name = Symbol(string(prefix) * "_constraints")
    quote
        function MOI.get(model::Optimizer, ::$attr,
                         c::MOI.ConstraintIndex{$function_type,$set_type})
            return model.$(array_name)[c.value].$attrfield
        end
    end
end

@define_get_con_attr(MOI.ConstraintSet,MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64},linear_le,set)
@define_get_con_attr(MOI.ConstraintSet,MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64},linear_ge,set)
@define_get_con_attr(MOI.ConstraintSet,MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64},linear_eq,set)
@define_get_con_attr(MOI.ConstraintSet,MOI.ScalarQuadraticFunction{Float64},MOI.LessThan{Float64},quadratic_le,set)
@define_get_con_attr(MOI.ConstraintSet,MOI.ScalarQuadraticFunction{Float64},MOI.GreaterThan{Float64},quadratic_ge,set)
@define_get_con_attr(MOI.ConstraintSet,MOI.ScalarQuadraticFunction{Float64},MOI.EqualTo{Float64},quadratic_eq,set)

@define_get_con_attr(MOI.ConstraintFunction,MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64},linear_le,func)
@define_get_con_attr(MOI.ConstraintFunction,MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64},linear_ge,func)
@define_get_con_attr(MOI.ConstraintFunction,MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64},linear_eq,func)


function MOI.get(model::Optimizer,::MOI.ConstraintSet,
                 c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}})
    return MOI.LessThan{Float64}(model.variable_info[c.value].upper_bound)
end

function MOI.get(model::Optimizer,::MOI.ConstraintSet,
                 c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}})
    return MOI.EqualTo{Float64}(model.variable_info[c.value].lower_bound)
end

function MOI.get(model::Optimizer,::MOI.ConstraintSet,
                 c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    return MOI.GreaterThan{Float64}(model.variable_info[c.value].lower_bound)
end
function MOI.get(model::Optimizer,::MOI.ConstraintFunction,
                 c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}})
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end
function MOI.get(model::Optimizer,::MOI.ConstraintFunction,
                 c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}})
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end
function MOI.get(
    model::Optimizer,::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end



function MOI.get(model::Optimizer, ::MOI.ListOfConstraints)
    constraints = Set{Tuple{DataType, DataType}}()

    for info in model.variable_info
        info.has_lower_bound && push!(constraints, (MOI.SingleVariable, MOI.LessThan{Float64}))
        info.has_upper_bound && push!(constraints, (MOI.SingleVariable, MOI.GreaterThan{Float64}))
        info.is_fixed && push!(constraints, (MOI.SingleVariable, MOI.EqualTo{Float64}))
    end

    isempty(model.linear_le_constraints) ||
        push!(constraints,(MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}))
    isempty(model.linear_ge_constraints) ||
        push!(constraints, (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}))
    isempty(model.linear_eq_constraints) ||
        push!(constraints, (MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}))
    isempty(model.quadratic_le_constraints) ||
        push!(constraints, (MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64}))
    isempty(model.quadratic_ge_constraints) ||
        push!(constraints, (MOI.ScalarQuadraticFunction{Float64}, MOI.GreaterThan{Float64}))
    isempty(model.quadratic_eq_constraints) ||
        push!(constraints, (MOI.ScalarQuadraticFunction{Float64}, MOI.EqualTo{Float64}))

    return collect(constraints)
end
function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
)
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}.(eachindex(model.linear_le_constraints))
end
function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
)
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}.(eachindex(model.linear_eq_constraints))
end
function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}
)
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}.(eachindex(model.linear_ge_constraints))
end
function MOI.get(model::Optimizer, ::MOI.ListOfConstraintIndices{MOI.SingleVariable, MOI.LessThan{Float64}})
    dict = Dict(model.variable_info[i] => i for i in 1:length(model.variable_info))
    filter!(info -> info.first.has_upper_bound, dict)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}.(values(dict))
end
function MOI.get(model::Optimizer, ::MOI.ListOfConstraintIndices{MOI.SingleVariable, MOI.EqualTo{Float64}})
    dict = Dict(model.variable_info[i] => i for i in 1:length(model.variable_info))
    filter!(info -> info.first.is_fixed, dict)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}.(values(dict))
end
function MOI.get(model::Optimizer, ::MOI.ListOfConstraintIndices{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    dict = Dict(model.variable_info[i] => i for i in 1:length(model.variable_info))
    filter!(info -> info.first.has_lower_bound, dict)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}.(values(dict))
end


function MOI.set(model::Optimizer, ::MOI.ConstraintSet,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}},
                 set::MOI.LessThan{Float64})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].upper_bound = set.upper
    return
end

function MOI.set(model::Optimizer, ::MOI.ConstraintSet,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}},
                 set::MOI.GreaterThan{Float64})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = set.lower
    return
end

function MOI.set(model::Optimizer, ::MOI.ConstraintSet,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}},
                 set::MOI.EqualTo{Float64})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = set.value
    model.variable_info[ci.value].upper_bound = set.value
    return
end

# default_copy_to
MOIU.supports_default_copy_to(model::Optimizer, copy_names::Bool) = !copy_names
MOI.copy_to(model::Optimizer, src::MOI.ModelLike; copy_names = false) = MOIU.default_copy_to(model, src, copy_names)

# objective sense
MOI.supports(::Optimizer,::MOI.ObjectiveSense) = true
MOI.set(model::Optimizer, ::MOI.ObjectiveSense,sense::MOI.OptimizationSense) = (model.sense = sense)
MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

# silence
const SILENT_KEY = :log_level
const SILENT_VAL = ERROR
MOI.supports(::Optimizer,::MOI.Silent) = true
MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)= value ?
    MOI.set(model,MOI.RawParameter(SILENT_KEY),SILENT_VAL) : delete!(model.option_dict,SILENT_KEY)
MOI.get(model::Optimizer, ::MOI.Silent) = get(model.option_dict,SILENT_KEY,String) == SILENT_VAL

# time limit
const TIME_LIMIT = :max_wall_time
MOI.supports(::Optimizer,::MOI.TimeLimitSec) = true
MOI.set(model::Optimizer,::MOI.TimeLimitSec,value)= value isa Real ?
    MOI.set(model,MOI.RawParameter(TIME_LIMIT),Float64(value)) : error("Invalid time limit: $value")
MOI.set(model::Optimizer,::MOI.TimeLimitSec,::Nothing)=delete!(model.option_dict,TIME_LIMIT)
MOI.get(model::Optimizer,::MOI.TimeLimitSec)=get(model.option_dict,TIME_LIMIT,Float64)

# set/get options
MOI.supports(::Optimizer,::MOI.RawParameter) = true
MOI.set(model::Optimizer, p::MOI.RawParameter, value) = (model.option_dict[Symbol(p.name)] = value)
MOI.get(model::Optimizer, p::MOI.RawParameter) = haskey(model.option_dict, Symbol(p.name)) ?
    (return model.option_dict[Symbol(p.name)]) : error("RawParameter with name $(p.name) is not set.")

# solve time
MOI.get(model::Optimizer, ::MOI.SolveTime) = model.ips.cnt.total_time

function MOI.empty!(model::Optimizer)
    # empty!(model.option_dict)
    empty!(model.variable_info)
    model.nlp_data = empty_nlp_data()
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective = nothing
    empty!(model.linear_le_constraints)
    empty!(model.linear_ge_constraints)
    empty!(model.linear_eq_constraints)
    empty!(model.quadratic_le_constraints)
    empty!(model.quadratic_ge_constraints)
    empty!(model.quadratic_eq_constraints)
    model.nlp_dual_start = nothing
    model.nlp = nothing
    model.ips = nothing
end

function MOI.is_empty(model::Optimizer)
    return isempty(model.variable_info) &&
        model.nlp_data.evaluator isa EmptyNLPEvaluator &&
        model.sense == MOI.FEASIBILITY_SENSE &&
        isempty(model.linear_le_constraints) &&
        isempty(model.linear_ge_constraints) &&
        isempty(model.linear_eq_constraints) &&
        isempty(model.quadratic_le_constraints) &&
        isempty(model.quadratic_ge_constraints) &&
        isempty(model.quadratic_eq_constraints)
end

function MOI.add_variable(model::Optimizer)
    push!(model.variable_info, VariableInfo())
    return MOI.VariableIndex(length(model.variable_info))
end
MOI.add_variables(model::Optimizer, n::Int) = [MOI.add_variable(model) for i in 1:n]

# checking inbounds
function check_inbounds(model::Optimizer, vi::MOI.VariableIndex)
    num_variables = length(model.variable_info)
    if !(1 <= vi.value <= num_variables)
        error("Invalid variable index $vi. ($num_variables variables in the model.)")
    end
end
check_inbounds(model::Optimizer, var::MOI.SingleVariable) = check_inbounds(model, var.variable)
function check_inbounds(model::Optimizer, aff::MOI.ScalarAffineFunction)
    for term in aff.terms
        check_inbounds(model, term.variable_index)
    end
end
function check_inbounds(model::Optimizer, quad::MOI.ScalarQuadraticFunction)
    for term in quad.affine_terms
        check_inbounds(model, term.variable_index)
    end
    for term in quad.quadratic_terms
        check_inbounds(model, term.variable_index_1)
        check_inbounds(model, term.variable_index_2)
    end
end

has_upper_bound(model::Optimizer, vi::MOI.VariableIndex) = model.variable_info[vi.value].has_upper_bound
has_lower_bound(model::Optimizer, vi::MOI.VariableIndex) = model.variable_info[vi.value].has_lower_bound
is_fixed(model::Optimizer, vi::MOI.VariableIndex) = model.variable_info[vi.value].is_fixed

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].upper_bound = Inf
    model.variable_info[ci.value].has_upper_bound = false
    return
end

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = -Inf
    model.variable_info[ci.value].has_lower_bound = false
    return
end

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}})
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = -Inf
    model.variable_info[ci.value].upper_bound = Inf
    model.variable_info[ci.value].is_fixed = false
    return
end


function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, lt::MOI.LessThan{Float64})
    vi = v.variable
    check_inbounds(model, vi)
    if isnan(lt.upper)
        error("Invalid upper bound value $(lt.upper).")
    end
    if has_upper_bound(model, vi)
        throw(MOI.UpperBoundAlreadySet{MOI.LessThan{Float64}, typeof(lt)}(vi))
    end
    if is_fixed(model, vi)
        throw(MOI.UpperBoundAlreadySet{MOI.EqualTo{Float64}, typeof(lt)}(vi))
    end
    model.variable_info[vi.value].upper_bound = lt.upper
    model.variable_info[vi.value].has_upper_bound = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, gt::MOI.GreaterThan{Float64})
    vi = v.variable
    check_inbounds(model, vi)
    if isnan(gt.lower)
        error("Invalid lower bound value $(gt.lower).")
    end
    if has_lower_bound(model, vi)
        throw(MOI.LowerBoundAlreadySet{MOI.GreaterThan{Float64}, typeof(gt)}(vi))
    end
    if is_fixed(model, vi)
        throw(MOI.LowerBoundAlreadySet{MOI.EqualTo{Float64}, typeof(gt)}(vi))
    end
    model.variable_info[vi.value].lower_bound = gt.lower
    model.variable_info[vi.value].has_lower_bound = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, eq::MOI.EqualTo{Float64})
    vi = v.variable
    check_inbounds(model, vi)
    if isnan(eq.value)
        error("Invalid fixed value $(gt.lower).")
    end
    if has_lower_bound(model, vi)
        throw(MOI.LowerBoundAlreadySet{MOI.GreaterThan{Float64}, typeof(eq)}(vi))
    end
    if has_upper_bound(model, vi)
        throw(MOI.UpperBoundAlreadySet{MOI.LessThan{Float64}, typeof(eq)}(vi))
    end
    if is_fixed(model, vi)
        throw(MOI.LowerBoundAlreadySet{MOI.EqualTo{Float64}, typeof(eq)}(vi))
    end
    model.variable_info[vi.value].lower_bound = eq.value
    model.variable_info[vi.value].upper_bound = eq.value
    model.variable_info[vi.value].is_fixed = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}(vi.value)
end

macro define_add_constraint(function_type, set_type, prefix)
    array_name = Symbol(string(prefix) * "_constraints")
    quote
        function MOI.add_constraint(model::Optimizer, func::$function_type, set::$set_type)
            check_inbounds(model, func)
            push!(model.$(array_name), ConstraintInfo(func, set))
            return MOI.ConstraintIndex{$function_type, $set_type}(length(model.$(array_name)))
        end
    end
end

@define_add_constraint(MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64},linear_le)
@define_add_constraint(MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}, linear_ge)
@define_add_constraint(MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64},linear_eq)
@define_add_constraint(MOI.ScalarQuadraticFunction{Float64},MOI.LessThan{Float64}, quadratic_le)
@define_add_constraint(MOI.ScalarQuadraticFunction{Float64},MOI.GreaterThan{Float64}, quadratic_ge)
@define_add_constraint(MOI.ScalarQuadraticFunction{Float64},MOI.EqualTo{Float64}, quadratic_eq)

function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart,
                 vi::MOI.VariableIndex, value::Union{Real, Nothing})
    check_inbounds(model, vi)
    model.variable_info[vi.value].start = value
    return
end

function MOI.supports(model::Optimizer, ::MOI.ConstraintDualStart,
                      ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}},
                      value::Union{Real, Nothing})
    return true
end
function MOI.set(model::Optimizer, ::MOI.ConstraintDualStart,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}},
                 value::Union{Real, Nothing})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    model.variable_info[vi.value].lower_bound_dual_start = value
    return
end
function MOI.supports(model::Optimizer, ::MOI.ConstraintDualStart,
                      ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}},
                      value::Union{Real, Nothing})
    return true
end
function MOI.set(model::Optimizer, ::MOI.ConstraintDualStart,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}},
                 value::Union{Real, Nothing})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    model.variable_info[vi.value].upper_bound_dual_start = value
    return
end
function MOI.supports(model::Optimizer, ::MOI.ConstraintDualStart,
                      ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}},
                      value::Union{Real, Nothing})
    return true
end
function MOI.set(model::Optimizer, ::MOI.ConstraintDualStart,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}},
                 value::Union{Real, Nothing})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    model.variable_info[vi.value].upper_bound_dual_start = value
    model.variable_info[vi.value].lower_bound_dual_start = value
    return
end

# NLPBlock Dual start
MOI.supports(::Optimizer, ::MOI.NLPBlockDualStart) = true
MOI.set(model::Optimizer, ::MOI.NLPBlockDualStart, values) = (model.nlp_dual_start = -values)
MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData) = (model.nlp_data = nlp_data)

function MOI.set(model::Optimizer, ::MOI.ObjectiveFunction,
                 func::Union{MOI.SingleVariable, MOI.ScalarAffineFunction,MOI.ScalarQuadraticFunction})
    check_inbounds(model, func)
    model.objective = func
    return
end

linear_le_offset(model::Optimizer) = 0
linear_ge_offset(model::Optimizer) = length(model.linear_le_constraints)
linear_eq_offset(model::Optimizer) = linear_ge_offset(model) + length(model.linear_ge_constraints)
quadratic_le_offset(model::Optimizer) = linear_eq_offset(model) + length(model.linear_eq_constraints)
quadratic_ge_offset(model::Optimizer) = quadratic_le_offset(model) + length(model.quadratic_le_constraints)
quadratic_eq_offset(model::Optimizer) = quadratic_ge_offset(model) + length(model.quadratic_ge_constraints)
nlp_constraint_offset(model::Optimizer) = quadratic_eq_offset(model) + length(model.quadratic_eq_constraints)

# Convenience functions used only in optimize!
function append_to_jacobian_sparsity!(I,J,
                                      aff::MOI.ScalarAffineFunction, row, offset)
    cnt = 0
    for term in aff.terms
        I[offset+cnt]= row
        J[offset+cnt]= term.variable_index.value
        cnt += 1
    end
    return cnt
end

function append_to_jacobian_sparsity!(I,J,
                                      quad::MOI.ScalarQuadraticFunction, row, offset)
    cnt = 0
    for term in quad.affine_terms
        I[offset+cnt]= row
        J[offset+cnt]= term.variable_index.value
        cnt += 1
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
        if row_idx == col_idx
            I[offset+cnt]= row
            J[offset+cnt]= row_idx.value
            cnt += 1
        else
            I[offset+cnt]= row
            J[offset+cnt]= row_idx.value
            I[offset+cnt+1]= row
            J[offset+cnt+1]= col_idx.value
            cnt += 2
        end
    end
    return cnt
end

# Refers to local variables in jacobian_structure() below.
macro append_to_jacobian_sparsity(array_name)
    escrow = esc(:row)
    escoffset = esc(:offset)
    quote
        for info in $(esc(array_name))
            $escoffset += append_to_jacobian_sparsity!($(esc(:I)), $(esc(:J)),info.func, $escrow, $escoffset)
            $escrow += 1
        end
    end
end

function jacobian_structure(model::Optimizer,I,J)
    offset = 1
    row = 1

    @append_to_jacobian_sparsity model.linear_le_constraints
    @append_to_jacobian_sparsity model.linear_ge_constraints
    @append_to_jacobian_sparsity model.linear_eq_constraints
    @append_to_jacobian_sparsity model.quadratic_le_constraints
    @append_to_jacobian_sparsity model.quadratic_ge_constraints
    @append_to_jacobian_sparsity model.quadratic_eq_constraints

    cnt = 0
    for (nlp_row, nlp_col) in MOI.jacobian_structure(model.nlp_data.evaluator)
        I[offset+cnt] = nlp_row + row - 1
        J[offset+cnt] = nlp_col
        cnt+=1
    end

    return I,J
end

append_to_hessian_sparsity!(I,J,::Union{MOI.SingleVariable,MOI.ScalarAffineFunction},offset) = 0

function append_to_hessian_sparsity!(I,J,quad::MOI.ScalarQuadraticFunction,offset)
    cnt = 0
    for term in quad.quadratic_terms
        I[offset+cnt]=term.variable_index_1.value
        J[offset+cnt]=term.variable_index_2.value
        cnt+=1
    end
    return cnt
end
function hessian_lagrangian_structure(model::Optimizer,I,J)
    offset = 1
    if !model.nlp_data.has_objective && model.objective !== nothing
        offset+=append_to_hessian_sparsity!(I,J,model.objective,offset)
    end
    for info in model.quadratic_le_constraints
        offset+=append_to_hessian_sparsity!(I,J,info.func,offset)
    end
    for info in model.quadratic_ge_constraints
        offset+=append_to_hessian_sparsity!(I,J,info.func,offset)
    end
    for info in model.quadratic_eq_constraints
        offset+=append_to_hessian_sparsity!(I,J,info.func,offset)
    end
    cnt = 0
    for (row,col) in MOI.hessian_lagrangian_structure(model.nlp_data.evaluator)
        I[offset+cnt]=row
        J[offset+cnt]=col
        cnt+=1
    end
    return I,J
end

get_nnz_hess(model::Optimizer) = get_nnz_hess_nonlinear(model) + get_nnz_hess_quadratic(model)
get_nnz_hess_nonlinear(model::Optimizer) =
    ((!model.nlp_data.has_objective && model.objective isa MOI.ScalarQuadraticFunction) ? length(model.objective.quadratic_terms) : 0) + length(MOI.hessian_lagrangian_structure(model.nlp_data.evaluator))
get_nnz_hess_quadratic(model::Optimizer) =
    (isempty(model.quadratic_eq_constraints) ? 0 : sum(length(term.func.quadratic_terms) for term in model.quadratic_eq_constraints)) +
    (isempty(model.quadratic_le_constraints) ? 0 : sum(length(term.func.quadratic_terms) for term in model.quadratic_le_constraints)) +
    (isempty(model.quadratic_ge_constraints) ? 0 : sum(length(term.func.quadratic_terms) for term in model.quadratic_ge_constraints))

get_nnz_jac(model::Optimizer) = get_nnz_jac_linear(model) + get_nnz_jac_quadratic(model) + get_nnz_jac_nonlinear(model)
get_nnz_jac_linear(model::Optimizer) =
    (isempty(model.linear_eq_constraints) ? 0 : sum(length(info.func.terms) for info in model.linear_eq_constraints)) +
    (isempty(model.linear_le_constraints) ? 0 : sum(length(info.func.terms) for info in model.linear_le_constraints)) +
    (isempty(model.linear_ge_constraints) ? 0 : sum(length(info.func.terms) for info in model.linear_ge_constraints))
get_nnz_jac_quadratic(model::Optimizer) =
    (isempty(model.quadratic_eq_constraints) ? 0 : sum(length(info.func.affine_terms) for info in model.quadratic_eq_constraints)) +
    (isempty(model.quadratic_eq_constraints) ? 0 : sum(isempty(info.func.quadratic_terms) ? 0 : sum(term.variable_index_1 == term.variable_index_2 ? 1 : 2 for term in info.func.quadratic_terms) for info in model.quadratic_eq_constraints)) +
    (isempty(model.quadratic_le_constraints) ? 0 : sum(length(info.func.affine_terms) for info in model.quadratic_le_constraints)) +
    (isempty(model.quadratic_le_constraints) ? 0 : sum(isempty(info.func.quadratic_terms) ? 0 : sum(term.variable_index_1 == term.variable_index_2 ? 1 : 2 for term in info.func.quadratic_terms) for info in model.quadratic_le_constraints)) +
    (isempty(model.quadratic_ge_constraints) ? 0 : sum(length(info.func.affine_terms) for info in model.quadratic_ge_constraints)) +
    (isempty(model.quadratic_ge_constraints) ? 0 : sum(isempty(info.func.quadratic_terms) ? 0 : sum(term.variable_index_1 == term.variable_index_2 ? 1 : 2 for term in info.func.quadratic_terms) for info in model.quadratic_ge_constraints))
get_nnz_jac_nonlinear(model::Optimizer) =
    (isempty(model.nlp_data.constraint_bounds) ? 0 : length(MOI.jacobian_structure(model.nlp_data.evaluator)))


function eval_function(var::MOI.SingleVariable, x)
    return x[var.variable.value]
end

function eval_function(aff::MOI.ScalarAffineFunction, x)
    function_value = aff.constant
    for term in aff.terms
        function_value += term.coefficient*x[term.variable_index.value]
    end
    return function_value
end

function eval_function(quad::MOI.ScalarQuadraticFunction, x)
    function_value = quad.constant
    for term in quad.affine_terms
        function_value += term.coefficient*x[term.variable_index.value]
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
        coefficient = term.coefficient
        if row_idx == col_idx
            function_value += 0.5*coefficient*x[row_idx.value]*x[col_idx.value]
        else
            function_value += coefficient*x[row_idx.value]*x[col_idx.value]
        end
    end
    return function_value
end

function eval_objective(model::Optimizer, x)
    # The order of the conditions is important. NLP objectives override regular
    # objectives.
    if model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    elseif model.objective !== nothing
        return eval_function(model.objective, x)
    else
        # No objective function set. This could happen with FEASIBILITY_SENSE.
        return 0.0
    end
end

function fill_gradient!(grad, x, var::MOI.SingleVariable)
    fill!(grad, 0.0)
    grad[var.variable.value] = 1.0
end

function fill_gradient!(grad, x, aff::MOI.ScalarAffineFunction{Float64})
    fill!(grad, 0.0)
    for term in aff.terms
        grad[term.variable_index.value] += term.coefficient
    end
end

function fill_gradient!(grad, x, quad::MOI.ScalarQuadraticFunction{Float64})
    fill!(grad, 0.0)
    for term in quad.affine_terms
        grad[term.variable_index.value] += term.coefficient
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
        coefficient = term.coefficient
        if row_idx == col_idx
            grad[row_idx.value] += coefficient*x[row_idx.value]
        else
            grad[row_idx.value] += coefficient*x[col_idx.value]
            grad[col_idx.value] += coefficient*x[row_idx.value]
        end
    end
end

function eval_objective_gradient(model::Optimizer, grad, x)
    if model.nlp_data.has_objective
        MOI.eval_objective_gradient(model.nlp_data.evaluator, grad, x)
    elseif model.objective !== nothing
        fill_gradient!(grad, x, model.objective)
    else
        fill!(grad, 0.0)
    end
    return
end

# Refers to local variables in eval_constraint() below.
macro eval_function(array_name)
    escrow = esc(:row)
    quote
        for info in $(esc(array_name))
            $(esc(:g))[$escrow] = eval_function(info.func, $(esc(:x)))
            $escrow += 1
        end
    end
end
function eval_constraint(model::Optimizer, g, x)
    row = 1
    @eval_function model.linear_le_constraints
    @eval_function model.linear_ge_constraints
    @eval_function model.linear_eq_constraints
    @eval_function model.quadratic_le_constraints
    @eval_function model.quadratic_ge_constraints
    @eval_function model.quadratic_eq_constraints
    nlp_g = view(g, row:length(g))
    MOI.eval_constraint(model.nlp_data.evaluator, nlp_g, x)
    return
end

function fill_constraint_jacobian!(values, start_offset, x, aff::MOI.ScalarAffineFunction)
    num_coefficients = length(aff.terms)
    for i in 1:num_coefficients
        values[start_offset+i] = aff.terms[i].coefficient
    end
    return num_coefficients
end

function fill_constraint_jacobian!(values, start_offset, x, quad::MOI.ScalarQuadraticFunction)
    num_affine_coefficients = length(quad.affine_terms)
    for i in 1:num_affine_coefficients
        values[start_offset+i] = quad.affine_terms[i].coefficient
    end
    num_quadratic_coefficients = 0
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
        coefficient = term.coefficient
        if row_idx == col_idx
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+1] = coefficient*x[col_idx.value]
            num_quadratic_coefficients += 1
        else
            # Note that the order matches the Jacobian sparsity pattern.
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+1] = coefficient*x[col_idx.value]
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+2] = coefficient*x[row_idx.value]
            num_quadratic_coefficients += 2
        end
    end
    return num_affine_coefficients + num_quadratic_coefficients
end

# Refers to local variables in eval_constraint_jacobian() below.
macro fill_constraint_jacobian(array_name)
    esc_offset = esc(:offset)
    quote
        for info in $(esc(array_name))
            $esc_offset += fill_constraint_jacobian!($(esc(:values)),
                                                     $esc_offset, $(esc(:x)),
                                                     info.func)
        end
    end
end

function eval_constraint_jacobian(model::Optimizer, values, x)
    offset = 0
    @fill_constraint_jacobian model.linear_le_constraints
    @fill_constraint_jacobian model.linear_ge_constraints
    @fill_constraint_jacobian model.linear_eq_constraints
    @fill_constraint_jacobian model.quadratic_le_constraints
    @fill_constraint_jacobian model.quadratic_ge_constraints
    @fill_constraint_jacobian model.quadratic_eq_constraints

    nlp_values = view(values, 1+offset:length(values))
    MOI.eval_constraint_jacobian(model.nlp_data.evaluator, nlp_values, x)
    return
end

function fill_hessian_lagrangian!(values, start_offset, scale_factor,
                                  ::Union{MOI.SingleVariable,
                                          MOI.ScalarAffineFunction,Nothing})
    return 0
end

function fill_hessian_lagrangian!(values, start_offset, scale_factor,
                                  quad::MOI.ScalarQuadraticFunction)
    for i in 1:length(quad.quadratic_terms)
        values[start_offset + i] = scale_factor*quad.quadratic_terms[i].coefficient
    end
    return length(quad.quadratic_terms)
end

function eval_hessian_lagrangian(model::Optimizer, values, x, obj_factor, lambda)
    offset = 0
    if !model.nlp_data.has_objective
        offset += fill_hessian_lagrangian!(values, 0, obj_factor,
                                           model.objective)
    end
    for (i, info) in enumerate(model.quadratic_le_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_le_offset(model)], info.func)
    end
    for (i, info) in enumerate(model.quadratic_ge_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_ge_offset(model)], info.func)
    end
    for (i, info) in enumerate(model.quadratic_eq_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_eq_offset(model)], info.func)
    end
    nlp_values = view(values, 1 + offset : length(values))
    nlp_lambda = view(lambda, 1 + nlp_constraint_offset(model) : length(lambda))
    MOI.eval_hessian_lagrangian(model.nlp_data.evaluator, nlp_values, x, obj_factor, nlp_lambda)
end



MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.nlp === nothing ?
    MOI.OPTIMIZE_NOT_CALLED : termination_status(model.nlp)
MOI.get(model::Optimizer, ::MOI.RawStatusString) = string(model.nlp.status)
MOI.get(model::Optimizer, ::MOI.ResultCount) = (model.nlp !== nothing) ? 1 : 0
MOI.get(model::Optimizer, attr::MOI.PrimalStatus) = !(1 <= attr.N <= MOI.get(model, MOI.ResultCount())) ?
    MOI.NO_SOLUTION : primal_status(model.ips)
MOI.get(model::Optimizer, attr::MOI.DualStatus) = !(1 <= attr.N <= MOI.get(model, MOI.ResultCount())) ?
    MOI.NO_SOLUTION : dual_status(model.ips)

const status_moi_dict = Dict(
    SOLVE_SUCCEEDED => MOI.LOCALLY_SOLVED,
    SOLVED_TO_ACCEPTABLE_LEVEL => MOI.ALMOST_LOCALLY_SOLVED,
    SEARCH_DIRECTION_BECOMES_TOO_SMALL => MOI.SLOW_PROGRESS,
    DIVERGING_ITERATES => MOI.INFEASIBLE_OR_UNBOUNDED,
    INFEASIBLE_PROBLEM_DETECTED => MOI.LOCALLY_INFEASIBLE,
    MAXIMUM_ITERATIONS_EXCEEDED => MOI.ITERATION_LIMIT,
    MAXIMUM_WALLTIME_EXCEEDED => MOI.TIME_LIMIT,
    INITIAL => MOI.OPTIMIZE_NOT_CALLED,
    RESTORATION_FAILED => MOI.NUMERICAL_ERROR,
    INVALID_NUMBER_DETECTED => MOI.INVALID_MODEL,
    ERROR_IN_STEP_COMPUTATION => MOI.NUMERICAL_ERROR,
    NOT_ENOUGH_DEGREES_OF_FREEDOM => MOI.INVALID_MODEL,
    USER_REQUESTED_STOP => MOI.INTERRUPTED,
    INTERNAL_ERROR => MOI.OTHER_ERROR)
const status_primal_dict = Dict(
    SOLVE_SUCCEEDED => MOI.FEASIBLE_POINT,
    SOLVED_TO_ACCEPTABLE_LEVEL => MOI.NEARLY_FEASIBLE_POINT,
    INFEASIBLE_PROBLEM_DETECTED => MOI.INFEASIBLE_POINT)
const status_dual_dict = Dict(
    SOLVE_SUCCEEDED => MOI.FEASIBLE_POINT,
    SOLVED_TO_ACCEPTABLE_LEVEL => MOI.NEARLY_FEASIBLE_POINT,
    INFEASIBLE_PROBLEM_DETECTED => MOI.INFEASIBLE_POINT)

termination_status(nlp::NonlinearProgram) = haskey(status_moi_dict,nlp.status) ?
    status_moi_dict[nlp.status] : MOI.UNKNOWN_RESULT_STATUS
termination_status(ips::Solver) = termination_status(ips.nlp)
primal_status(ips::Solver) = haskey(status_primal_dict,ips.nlp.status) ?
    status_primal_dict[ips.nlp.status] : MOI.UNKNOWN_RESULT_STATUS
dual_status(ips::Solver) = haskey(status_dual_dict,ips.nlp.status) ?
    status_dual_dict[ips.nlp.status] : MOI.UNKNOWN_RESULT_STATUS


function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    scale = (model.sense == MOI.MAX_SENSE) ? -1 : 1
    return scale * model.nlp.obj_val
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    check_inbounds(model, vi)
    return model.nlp.x[vi.value]
end

macro define_constraint_primal(function_type, set_type, prefix)
    constraint_array = Symbol(string(prefix) * "_constraints")
    offset_function = Symbol(string(prefix) * "_offset")
    quote
        function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal,
                         ci::MOI.ConstraintIndex{$function_type, $set_type})
            MOI.check_result_index_bounds(model, attr)
            if !(1 <= ci.value <= length(model.$(constraint_array)))
                error("Invalid constraint index ", ci.value)
            end
            return model.ips.c[ci.value + $offset_function(model)]
        end
    end
end

@define_constraint_primal(MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}, linear_le)
@define_constraint_primal(MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}, linear_ge)
@define_constraint_primal(MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}, linear_eq)
@define_constraint_primal(MOI.ScalarQuadraticFunction{Float64},MOI.LessThan{Float64}, quadratic_le)
@define_constraint_primal(MOI.ScalarQuadraticFunction{Float64},MOI.GreaterThan{Float64}, quadratic_ge)
@define_constraint_primal(MOI.ScalarQuadraticFunction{Float64},MOI.EqualTo{Float64}, quadratic_eq)

# function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal,
#                  ci::MOI.ConstraintIndex{MOI.SingleVariable,
#                                          MOI.LessThan{Float64}})
#     MOI.check_result_index_bounds(model, attr)
#     vi = MOI.VariableIndex(ci.value)
#     check_inbounds(model, vi)
#     if !has_upper_bound(model, vi)
#         error("Variable $vi has no upper bound -- ConstraintPrimal not defined.")
#     end
#     return model.nlp.x[vi.value]
# end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         MOI.GreaterThan{Float64}})
    MOI.check_result_index_bounds(model, attr)
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    if !has_lower_bound(model, vi)
        error("Variable $vi has no lower bound -- ConstraintPrimal not defined.")
    end
    return model.nlp.x[vi.value]
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         MOI.EqualTo{Float64}})
    MOI.check_result_index_bounds(model, attr)
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    if !is_fixed(model, vi)
        error("Variable $vi is not fixed -- ConstraintPrimal not defined.")
    end
    return model.nlp.x[vi.value]
end

macro define_constraint_dual(function_type, set_type, prefix)
    constraint_array = Symbol(string(prefix) * "_constraints")
    offset_function = Symbol(string(prefix) * "_offset")
    quote
        function MOI.get(model::Optimizer, attr::MOI.ConstraintDual,
                         ci::MOI.ConstraintIndex{$function_type, $set_type})
            MOI.check_result_index_bounds(model, attr)
            if !(1 <= ci.value <= length(model.$(constraint_array)))
                error("Invalid constraint index ", ci.value)
            end
            return -1 * model.nlp.l[ci.value + $offset_function(model)]
        end
    end
end
@define_constraint_dual(MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}, linear_le)
@define_constraint_dual(MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}, linear_ge)
@define_constraint_dual(MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}, linear_eq)
@define_constraint_dual(MOI.ScalarQuadraticFunction{Float64},MOI.LessThan{Float64}, quadratic_le)
@define_constraint_dual(MOI.ScalarQuadraticFunction{Float64},MOI.GreaterThan{Float64}, quadratic_ge)
@define_constraint_dual(MOI.ScalarQuadraticFunction{Float64},MOI.EqualTo{Float64}, quadratic_eq)

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}})
    MOI.check_result_index_bounds(model, attr)
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    has_upper_bound(model, vi) || error("Variable $vi has no upper bound -- ConstraintDual not defined.")
    return -1 * model.nlp.zu[vi.value] # MOI convention is for feasible LessThan duals to be nonpositive.
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         MOI.GreaterThan{Float64}})
    MOI.check_result_index_bounds(model, attr)
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    has_lower_bound(model, vi) || error("Variable $vi has no lower bound -- ConstraintDual not defined.")
    return model.nlp.zl[vi.value]
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         MOI.EqualTo{Float64}})
    MOI.check_result_index_bounds(model, attr)
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    if !is_fixed(model, vi)
        error("Variable $vi is not fixed -- ConstraintDual not defined.")
    end
    return model.nlp.zl[vi.value] - model.nlp.zu[vi.value]
end

function MOI.get(model::Optimizer, attr::MOI.NLPBlockDual)
    MOI.check_result_index_bounds(model, attr)
    return -1 * model.nlp.l[(1 + nlp_constraint_offset(model)):end]
end

function num_constraints(model::Optimizer)
    m_linear_le = length(model.linear_le_constraints)
    m_linear_ge = length(model.linear_ge_constraints)
    m_linear_eq = length(model.linear_eq_constraints)
    m_quadratic_le = length(model.quadratic_le_constraints)
    m_quadratic_ge = length(model.quadratic_ge_constraints)
    m_quadratic_eq = length(model.quadratic_eq_constraints)
    m_nl = length(model.nlp_data.constraint_bounds)
    return m_linear_le + m_linear_ge + m_linear_eq + m_quadratic_le + m_quadratic_ge + m_quadratic_eq + m_nl
end

function is_jac_hess_constant(model::Optimizer)
    isempty(model.nlp_data.constraint_bounds) || return (false,false)
    isempty(model.quadratic_le_constraints) || return (false,false)
    isempty(model.quadratic_ge_constraints) || return (false,false)
    isempty(model.quadratic_eq_constraints) || return (false,false)
    return (true,model.nlp_data.has_objective ? false : true)
end

zero_if_nothing(x) = x == nothing ? 0. : x

function set_x!(model,x,xl,xu,zl,zu)

    for i=1:length(model.variable_info)
        info = model.variable_info[i]
        x[i]  = info.start == nothing ? 0. : info.start
        xl[i] = info.lower_bound
        xu[i] = info.upper_bound
        zl[i] = zero_if_nothing(info.lower_bound_dual_start)
        zu[i] = zero_if_nothing(info.upper_bound_dual_start)
        i += 1
    end
end

function set_g!(model::Optimizer,l,gl,gu)
    i = 1
    for info in model.linear_le_constraints
        l[i] = zero_if_nothing(info.dual_start)
        gl[i]= -Inf
        gu[i]= info.set.upper
        i+=1
    end
    for info in model.linear_ge_constraints
        l[i] = zero_if_nothing(info.dual_start)
        gl[i]= info.set.lower
        gu[i]= Inf
        i+=1
    end
    for info in model.linear_eq_constraints
        l[i] = zero_if_nothing(info.dual_start)
        gl[i]= info.set.value
        gu[i]= info.set.value
        i+=1
    end
    for info in model.quadratic_le_constraints
        l[i] = zero_if_nothing(info.dual_start)
        gl[i]= -Inf
        gu[i]= info.set.upper
        i+=1
    end
    for info in model.quadratic_ge_constraints
        l[i] = zero_if_nothing(info.dual_start)
        gl[i]= info.set.lower
        gu[i]= Inf
        i+=1
    end
    for info in model.quadratic_eq_constraints
        l[i] = zero_if_nothing(info.dual_start)
        gl[i]= info.set.value
        gu[i]= info.set.value
        i+=1
    end
    j=0
    for bound in model.nlp_data.constraint_bounds
        gl[i+j]= bound.lower
        gu[i+j]= bound.upper
        j+=1
    end
    k=0
    if model.nlp_dual_start != nothing
        for v in model.nlp_dual_start
            l[i+k] = v
            k+=1
        end
    else
        l[i:end] .= 0.
    end
end

num_variables(model::Optimizer) = length(model.variable_info)

function get_obj_scale(sense)
    sense == MOI.MIN_SENSE && return 1.
    sense == MOI.MAX_SENSE && return -1.
    return 0.
end

function NonlinearProgram(model::Optimizer)
    :Hess in MOI.features_available(model.nlp_data.evaluator) || error("Hessian information is needed.")
    MOI.initialize(model.nlp_data.evaluator, [:Grad,:Hess,:Jac])

    n = num_variables(model)
    m = num_constraints(model)
    nnz_hess = get_nnz_hess(model)
    nnz_jac = get_nnz_jac(model)

    x  = Vector{Float64}(undef,n)
    g  = Vector{Float64}(undef,m)
    xl = Vector{Float64}(undef,n)
    xu = Vector{Float64}(undef,n)
    zl = Vector{Float64}(undef,n)
    zu = Vector{Float64}(undef,n)

    l = Vector{Float64}(undef,m)
    gl = Vector{Float64}(undef,m)
    gu = Vector{Float64}(undef,m)

    set_x!(model,x,xl,xu,zl,zu)
    set_g!(model,l,gl,gu)

    obj_scale = get_obj_scale(model.sense)
    obj(x::AbstractArray{Float64,1}) = obj_scale*eval_objective(model,x)
    obj_grad!(f::AbstractArray{Float64,1},x::AbstractArray{Float64,1}) =
        (eval_objective_gradient(model,f,x); obj_scale!=1. && (f.*=obj_scale))
    con!(c::Array{Float64,1},x::AbstractArray{Float64,1}) = eval_constraint(model,c,x)
    con_jac!(jac::AbstractArray{Float64,1},
             x::AbstractArray{Float64,1})=eval_constraint_jacobian(model,jac,x)
    lag_hess!(hess::AbstractArray{Float64,1},x::AbstractArray{Float64,1},l::AbstractArray{Float64,1},
              sig::Float64) = eval_hessian_lagrangian(model,hess,x,obj_scale*sig,l)
    hess_sparsity!(I,J)= hessian_lagrangian_structure(model,I,J)
    jac_sparsity!(I,J) = jacobian_structure(model,I,J)

    model.option_dict[:jacobian_constant], model.option_dict[:hessian_constant] = is_jac_hess_constant(model)
    model.option_dict[:dual_initialized] = !iszero(l)

    return NonlinearProgram(n,m,nnz_hess,nnz_jac,0.,x,g,l,zl,zu,xl,xu,gl,gu,obj,obj_grad!,con!,con_jac!,
                            lag_hess!,hess_sparsity!,jac_sparsity!,INITIAL,Dict{Symbol,Any}())
end

function MOI.optimize!(model::Optimizer)
    model.nlp = NonlinearProgram(model)
    model.ips = Solver(model.nlp;option_dict=copy(model.option_dict))
    optimize!(model.ips)
    model.solve_time = model.ips.cnt.total_time
    return
end
