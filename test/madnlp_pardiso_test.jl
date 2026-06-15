# PardisoMKL integration tests, incorporated from the old lib/MadNLPPardiso/test.
# PardisoMKLSolver builds on MKL_jll (via MadCorePardiso, reexported by MadNLP).
# Intel MKL's Pardiso is reliably functional only on x86 Linux here (it throws in
# pardisoinit on the ARM macOS runners and is flaky on Windows), so guard on Linux
# — matching where MKL Pardiso has historically been tested. The licensed
# PardisoSolver is omitted (its license has expired), matching the original suite.

@testset "PardisoMKL" begin
    if Sys.islinux()
        MadNLPTests.test_linear_solver(PardisoMKLSolver, Float32)
        MadNLPTests.test_linear_solver(PardisoMKLSolver, Float64)

        test_madnlp(
            "PardisoMKL",
            () -> MadNLP.Optimizer(linear_solver = PardisoMKLSolver, print_level = MadNLP.ERROR),
            ["eigmina"],
        )
    else
        @info "Skipping PardisoMKL tests (MKL Pardiso not functional on this platform)"
        @test_skip true
    end
end
