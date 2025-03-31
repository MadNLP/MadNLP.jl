using Test, MadNLP, MadNLPMumps, MadNLPTests, MPI

testset = [
    [
        "Mumps",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMumps.MumpsSolver,
            print_level=MadNLP.ERROR),
        []
    ],
]

@testset "MadNLPMumps test" begin
    MPI.Init() # TODO: check if we really need MPI
    comm = MPI.COMM_WORLD
    MadNLPTests.test_linear_solver(MadNLPMumps.MumpsSolver,Float32)
    MadNLPTests.test_linear_solver(MadNLPMumps.MumpsSolver,Float64)
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude)
    end
end

