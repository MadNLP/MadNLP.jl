using Test, MadNLP, MadNLPHSL, MadNLPTests

testset = [
    [
        "HSL-Ma27",
        ()->MadNLP.Optimizer(
            linear_solver=Ma27Solver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "HSL-Ma57",
        ()->MadNLP.Optimizer(
            linear_solver=Ma57Solver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "HSL-Ma77",
        ()->MadNLP.Optimizer(
            linear_solver=Ma77Solver,
            print_level=MadNLP.ERROR),
        ["unbounded"]
    ],
    [
        "HSL-Ma86",
        ()->MadNLP.Optimizer(
            linear_solver=Ma86Solver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "HSL-Ma97",
        ()->MadNLP.Optimizer(
            linear_solver=Ma97Solver,
            print_level=MadNLP.ERROR),
        []
    ],
]

@testset "MadNLPHSL test" begin
    for hsl_solver in [Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver]
        MadNLPTests.test_linear_solver(hsl_solver)
    end
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude)
    end
end


