using Test, MadNLP, MadNLPCliqueTrees, MadNLPTests
using SparseArrays

@testset "MadNLPCliqueTrees test" begin
    # Default LDL algorithm
    MadNLPTests.test_linear_solver(CliqueTreesSolver, Float32)
    MadNLPTests.test_linear_solver(CliqueTreesSolver, Float64)

    # Cholesky algorithm on positive definite matrix
    @testset "Cholesky algorithm" begin
        for T in (Float32, Float64)
            row = Int32[1,2,2]; col = Int32[1,1,2]; val = T[1., .1, 2.]
            b = T[1.0, 3.0]
            sol = T[0.8542713567839195, 1.4572864321608041]
            csc = sparse(row, col, val, 2, 2)
            opt = CliqueTreesOptions(cliquetrees_algorithm=MadNLP.CHOLESKY)
            M = CliqueTreesSolver(csc; opt=opt)
            MadNLP.factorize!(M)
            @test MadNLP.is_inertia(M)
            @test MadNLP.inertia(M) == (2, 0, 0)
            x = MadNLP.solve_linear_system!(M, copy(b))
            @test MadNLPTests.solcmp(x, sol)
        end
    end

    test_madnlp(
        "CliqueTrees",
        () -> MadNLP.Optimizer(linear_solver = CliqueTreesSolver, print_level = MadNLP.ERROR),
        [],
    )
end
