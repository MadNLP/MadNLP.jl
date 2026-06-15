# A no-op linear solver. It satisfies the AbstractLinearSolver interface but does
# nothing — `solve_linear_system!` leaves the right-hand side untouched. It is
# MadNLPCore's `default_sparse_solver`, so the bare interior-point solver loads
# and constructs without depending on any real sparse-solver backend. To actually
# solve, pass a real `linear_solver` (e.g. `MumpsSolver` from MadCoreMUMPS) or use
# MadNLP, which defaults `default_sparse_solver` to `MumpsSolver`.

@kwdef mutable struct DummyOptions <: AbstractOptions end

mutable struct DummyLinearSolver{T} <: AbstractLinearSolver{T}
    opt::DummyOptions
    logger::MadNLPLogger
end

function DummyLinearSolver(A; opt = DummyOptions(), logger = MadNLPLogger())
    return DummyLinearSolver{eltype(A)}(opt, logger)
end

default_options(::Type{DummyLinearSolver}) = DummyOptions()
input_type(::Type{DummyLinearSolver}) = :csc
is_supported(::Type{DummyLinearSolver}, ::Type{T}) where {T <: AbstractFloat} = true
introduce(::DummyLinearSolver) = "Dummy (no-op) linear solver"
factorize!(M::DummyLinearSolver) = M
solve_linear_system!(::DummyLinearSolver, x::AbstractVector) = x
is_inertia(::DummyLinearSolver) = false
improve!(::DummyLinearSolver) = false
