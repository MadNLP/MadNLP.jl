using Test, MadNLP, MadNLPPardiso, MadNLPTests

testset = [
    # TODO; Pardiso license has expired
    # [
    #     "Pardiso",
    #     ()->MadNLP.Optimizer(
    #         linear_solver=MadNLPPardiso,
    #         print_level=MadNLP.ERROR),
    #     [],
    #     @isdefined(MadNLPPardiso)
    # ],
    [
        "PardisoMKL",
        ()->MadNLP.Optimizer(
            linear_solver=PardisoMKLSolver,
            print_level=MadNLP.ERROR),
        ["eigmina"]
    ]
]

@testset "MadNLPPardiso test" begin

    MadNLPTests.test_linear_solver(PardisoMKLSolver,Float32)
    MadNLPTests.test_linear_solver(PardisoMKLSolver,Float64)
    # MadNLPTests.test_linear_solver(PardisoSolver)
    
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude)
    end
end


