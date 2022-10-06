module TestMOIWrapper

using MadNLP
using Test

const MOI = MadNLP.MathOptInterface

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
        exclude = String[
            "test_modification",
            "test_attribute_TimeLimitSec",
            # TODO: MadNLP does not return the correct multiplier
            # when a variable is fixed with MOI.EqualTo.
            "test_linear_integration",
            "test_quadratic_constraint_GreaterThan",
            "test_quadratic_constraint_LessThan",
            # MadNLP reaches maximum number of iterations instead
            # of returning infeasibility certificate.
            "test_linear_DUAL_INFEASIBLE",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            # Tests excluded on purpose
            # - Excluded because Hessian information is needed
            "test_nonlinear_hs071_hessian_vector_product",
            # - Excluded because Hessian information is needed
            "test_nonlinear_hs071_no_hessian",
            # - Excluded because Hessian information is needed
            "test_nonlinear_invalid",

            #  - Excluded because this test is optional
            "test_model_ScalarFunctionConstantNotZero",
        ]
    )

    return
end

function test_Name()
    model = MadNLP.Optimizer()
    @test MOI.supports(model, MOI.Name())
    @test MOI.get(model, MOI.Name()) == ""
    MOI.set(model, MOI.Name(), "Model")
    @test MOI.get(model, MOI.Name()) == "Model"
    return
end

end

TestMOIWrapper.runtests()
