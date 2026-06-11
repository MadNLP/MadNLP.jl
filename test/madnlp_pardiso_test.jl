# PardisoMKL integration tests, incorporated from the old lib/MadNLPPardiso/test.
# PardisoMKLSolver builds on the freely-available MKL_jll (via MadCorePardiso,
# reexported by MadNLP), so this runs on any runner. The licensed PardisoSolver is
# omitted (its license has expired), matching the original suite.

@testset "PardisoMKL" begin
    MadNLPTests.test_linear_solver(PardisoMKLSolver, Float32)
    MadNLPTests.test_linear_solver(PardisoMKLSolver, Float64)

    test_madnlp(
        "PardisoMKL",
        () -> MadNLP.Optimizer(linear_solver = PardisoMKLSolver, print_level = MadNLP.ERROR),
        ["eigmina"],
    )
end
