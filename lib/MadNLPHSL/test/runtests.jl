using Test, MadNLP, MadNLPHSL, MadNLPTests

testset = [
    [
        "HSL-Ma27",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMa27,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "HSL-Ma57",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMa57,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "HSL-Ma77",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMa77,
            print_level=MadNLP.ERROR),
        ["unbounded"]
    ],
    [
        "HSL-Ma86",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMa86,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "HSL-Ma97",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMa97,
            print_level=MadNLP.ERROR),
        []
    ],
]

@testset "MadNLPHSL test" begin
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude)
    end
end


