using Test, MadCoreLDLFactorizations, SparseArrays
import MadCore:
    default_options, improve!, factorize!, solve_linear_system!,
    is_inertia, inertia, introduce

# Standalone linear-solver test (depends on MadCore only): factorize + solve a
# small SPD system, mirroring MadCoreHSL's test_solver.
function test_solver(S, T)
    csc = sparse(Int32[1, 2, 2], Int32[1, 1, 2], T[1.0, 0.1, 2.0], 2, 2)
    sol = T[0.8542713567839195, 1.4572864321608041]
    M = S(csc; opt = default_options(S))
    introduce(M)
    improve!(M)
    factorize!(M)
    is_inertia(M) && @test inertia(M) == (2, 0, 0)
    x = solve_linear_system!(M, T[1.0, 3.0])
    return @test isapprox(x, sol; atol = 1.0e-6)
end

@testset "MadCoreLDLFactorizations" begin
    test_solver(LDLSolver, Float64)
end
