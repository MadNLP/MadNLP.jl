using Test, MadNLP, MadNLPKrylov, MadNLPTests

testset = [
    [
        "Iterative",
        ()->MadNLP.Optimizer(
            iterative_solver=MadNLPKrylov,
            print_level=MadNLP.ERROR),
        []
    ],
]

@testset "MadNLPKrylov test" begin
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude)
    end
end


