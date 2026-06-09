using Test, MadCoreHSL, SparseArrays
import MadCore:
    default_options, improve!, factorize!, solve_linear_system!,
    is_inertia, inertia, introduce, input_type

# Standalone linear-solver test (no MadNLP/IPM dependency): factorize and solve a
# small symmetric-indefinite system, mirroring MadNLPTests.test_linear_solver.
# Requires a licensed HSL_jll to be available (dev it locally; never committed).
function test_solver(S, T)
    row = Int32[1, 2, 2]
    col = Int32[1, 1, 2]
    val = T[1.0, 0.1, 2.0]
    b   = T[1.0, 3.0]
    sol = T[0.8542713567839195, 1.4572864321608041]
    csc = sparse(row, col, val, 2, 2)
    M = S(csc; opt = default_options(S))
    introduce(M)
    improve!(M)
    factorize!(M)
    is_inertia(M) && @test inertia(M) == (2, 0, 0)
    x = solve_linear_system!(M, copy(b))
    @test isapprox(x, sol; atol = 1e-6)
end

@testset "MadCoreHSL" begin
    @testset "$(nameof(S))" for S in
        [Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver]
        test_solver(S, Float64)
    end
end
