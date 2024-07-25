module TestMOIWrapper

using MadNLP
using Test

using MathOptInterface
const MOI = MathOptInterface

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_MOI_Test()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MadNLP.Optimizer(),
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            atol = 1e-4,
            rtol = 1e-4,
            infeasible_status = MOI.LOCALLY_INFEASIBLE,
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
                MOI.ObjectiveBound,
            ]
        );
        exclude = [
            # MadNLP reaches maximum number of iterations instead
            # of returning infeasibility certificate.
            r"test_linear_DUAL_INFEASIBLE.*",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            # Tests excluded on purpose
            # - Excluded because Hessian information is needed
            "test_nonlinear_hs071_hessian_vector_product",
            # - Excluded because Hessian information is needed
            "test_nonlinear_invalid",

            #  - Excluded because this test is optional
            "test_model_ScalarFunctionConstantNotZero",
            # Throw an error: "Unable to query the dual of a variable
            # bound that was reformulated using `ZerosBridge`."
            "test_linear_VectorAffineFunction_empty_row",
            "test_conic_linear_VectorOfVariables_2",
            # TODO: investigate why it is breaking.
            "test_nonlinear_expression_hs109",
        ]
    )

    return
end

function test_extra()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("linear_solver"), UmfpackSolver)

    @test MOI.supports(model, MOI.Name())
    @test MOI.get(model, MOI.Name()) == ""
    MOI.set(model, MOI.Name(), "Model")
    @test MOI.get(model, MOI.Name()) == "Model"

    @test MOI.get(model, MOI.BarrierIterations()) == 0

    return
end

# See issue #239 (https://github.com/MadNLP/MadNLP.jl/issues/239)
function test_invalid_number_in_hessian_lagrangian()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    nlp = MOI.Nonlinear.Model()
    MOI.Nonlinear.set_objective(nlp, :(($x - 5)^2 + ($y - 8)^2))
    MOI.Nonlinear.add_constraint(nlp, :($x * $y), MOI.EqualTo(5.0))
    ev = MOI.Nonlinear.Evaluator(nlp, MOI.Nonlinear.SparseReverseMode(), [x, y])
    MOI.set(model, MOI.NLPBlock(), MOI.NLPBlockData(ev))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
    return
end

# See issue #318
function test_user_defined_function()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    # Define custom function.
    f(a, b) = a^2 + b^2
    x = MOI.add_variables(model, 2)
    MOI.set(model, MOI.UserDefinedFunction(:f, 2), (f,))
    obj_f = MOI.ScalarNonlinearFunction(:f, Any[x[1], x[2]])
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj_f)}(), obj_f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
end

end

TestMOIWrapper.runtests()
