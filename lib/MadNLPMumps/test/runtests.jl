using Test, MadNLP, MadNLPMumps, MadNLPTests

testset = [
    [
        "Mumps",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMumps,
            print_level=MadNLP.ERROR),
        []
    ],
]

@testset "MadNLPMumps test" begin
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude)
    end
end


