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
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            MadNLP.Optimizer(),
        ),
        Float64,
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            atol = 1e-4,
            rtol = 1e-4,
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.delete,
                MOI.ConstraintDual,
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
                MOI.ObjectiveBound,
            ]
        );
        exclude = String[
            "test_modification",
            # - Need to implement TimeLimitSec
            "test_attribute_TimeLimitSec",
            # - Wrong return type
            "test_model_UpperBoundAlreadySet",
            # - Final objective value is not equal to 0.0
            "test_objective_FEASIBILITY_SENSE_clears_objective",

            # TODO: Need to investigate why these tests are breaking
            #   get(model, MOI.ConstraintPrimal(), c) returns the
            #   opposite value: if 1.0 is expected, -1.0 is returned
            "test_constraint_ScalarAffineFunction_EqualTo",
            "test_quadratic_nonconvex_constraint_basic",
            "test_linear_integration",

            # TODO: there might be an issue with VectorAffineFunction/VectorOfVariables
            "test_conic_NormOneCone_VectorOfVariables",
            "test_conic_NormOneCone_VectorAffineFunction",
            "test_conic_NormInfinityCone_VectorOfVariables",
            "test_conic_NormInfinityCone_VectorAffineFunction",
            "test_conic_linear_VectorAffineFunction",
            "test_conic_linear_VectorOfVariables",

            # Tests excluded on purpose
            # - Excluded as MadNLP returns LOCALLY_INFEASIBLE instead of INFEASIBLE
            "INFEASIBLE",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_",
            # - Excluded because Hessian information is needed
            "test_nonlinear_hs071_hessian_vector_product",
            # - Excluded because Hessian information is needed
            "test_nonlinear_hs071_no_hessian",
            # - Excluded because Hessian information is needed
            "test_nonlinear_invalid",

            #  - Excluded because this test is optional
            "test_model_ScalarFunctionConstantNotZero",
            #  - Excluded because MadNLP returns INVALID_MODEL instead of LOCALLY_SOLVED
            "test_linear_VectorAffineFunction_empty_row",
        ]
    )
    return
end

end

TestMOIWrapper.runtests()
