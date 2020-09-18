import MadNLP
import MINLPTests

using Test

const OPTIMIZER = MadNLP.Optimizer

@testset "MINLPTests" begin
    ###
    ### src/nlp tests.
    ###

    MINLPTests.test_nlp(
        OPTIMIZER,
        #exclude = [
        #    "005_011",  # Uses the function `\`
        #    "006_010",  # User-defined function
        #    "007_010",  # Infeasible model
        #],
        objective_tol = 1e-5,
        primal_tol = 1e-5,
        dual_tol = NaN,
    )

    #@testset "nlp_007_010" begin
    #    MINLPTests.nlp_007_010(
    #        OPTIMIZER,
    #        1e-5,
    #        NaN,
    #        NaN,
    #        Dict(MINLPTests.INFEASIBLE_PROBLEM => MadNLP.MOI.INFEASIBLE),
    #        Dict(MINLPTests.INFEASIBLE_PROBLEM => MadNLP.MOI.NO_SOLUTION),
    #    )
    #end

    ###
    ### src/nlp-cvx tests.
    ###

    MINLPTests.test_nlp_cvx(
        OPTIMIZER,
        #exclude = [
        #    "109_010"  # Ipopt fails to converge
        #]
    )
end


