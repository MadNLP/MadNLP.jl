using Test, MadCorePardiso, SparseArrays
import MadCore:
    default_options, improve!, factorize!, solve_linear_system!, introduce

# Factorize + solve a small symmetric-indefinite system, returning the solution
# (throws if the backend library is unavailable). Standalone — no MadNLP/IPM dep.
function solve_system(S, T)
    row = Int32[1, 2, 2]
    col = Int32[1, 1, 2]
    val = T[1.0, 0.1, 2.0]
    b = T[1.0, 3.0]
    csc = sparse(row, col, val, 2, 2)
    M = S(csc; opt = default_options(S))
    introduce(M)
    improve!(M)
    factorize!(M)
    return solve_linear_system!(M, copy(b))
end

const SOL = [0.8542713567839195, 1.4572864321608041]

@testset "MadCorePardiso" begin
    # PardisoMKLSolver uses MKL_jll, which is freely available.
    @testset "PardisoMKLSolver" begin
        @test isapprox(solve_system(PardisoMKLSolver, Float64), SOL; atol = 1.0e-6)
    end
    # PardisoSolver wraps the proprietary Panua PARDISO library (point
    # JULIA_PARDISO at its folder). Skip it when unset: not only is the library
    # absent, but with an empty libpardiso path its ccalls would fall through to
    # MKL's `pardiso` symbol and silently "pass", which would be misleading.
    @testset "PardisoSolver (proprietary Panua)" begin
        if haskey(ENV, "JULIA_PARDISO")
            @test isapprox(solve_system(PardisoSolver, Float64), SOL; atol = 1.0e-6)
        else
            @test_skip isapprox(solve_system(PardisoSolver, Float64), SOL; atol = 1.0e-6)
        end
    end
end
