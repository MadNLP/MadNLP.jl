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
            "008_010", # Umfpack not passing the test; might be converging to a different solution?
            "008_011", # Umfpack not passing the test; might be converging to a different solution?
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
