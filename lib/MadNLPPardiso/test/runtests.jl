using Test, MadNLP, MadNLPPardiso, MadNLPTests

testset = [
    [
        "Pardiso",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPPardiso,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPPardiso)
    ],
]

@testset "MadNLPPardiso test" begin
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude)
    end
end


