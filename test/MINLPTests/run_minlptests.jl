import MadNLP
import MINLPTests

using Test

const OPTIMIZER = ()->MadNLP.Optimizer(print_level=MadNLP.ERROR)

@testset "MINLPTests" begin
    ###
    ### src/nlp tests.
    ###

    MINLPTests.test_nlp(
        OPTIMIZER,
        exclude = [
            "005_011",  # Uses the function `\`
            "006_010",  # User-defined function without Hessian (autodiff only provides 1st order)
        ],
        objective_tol = 1e-5,
        primal_tol = 1e-5,
        dual_tol = NaN,
    )

    ###
    ### src/nlp-cvx tests.
    ###

    MINLPTests.test_nlp_cvx(
        OPTIMIZER,
    )
end


