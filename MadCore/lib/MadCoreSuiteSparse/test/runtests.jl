using Test, MadCoreSuiteSparse, SparseArrays
import MadCore:
    default_options, improve!, factorize!, solve_linear_system!,
    is_inertia, inertia, introduce

const SOL = [0.8542713567839195, 1.4572864321608041]

# CHOLMOD is a symmetric Cholesky solver: pass the lower triangle.
function test_symmetric_solver(S, T)
    csc = sparse(Int32[1, 2, 2], Int32[1, 1, 2], T[1.0, 0.1, 2.0], 2, 2)
    M = S(csc; opt = default_options(S))
    introduce(M)
    improve!(M)
    factorize!(M)
    is_inertia(M) && @test inertia(M) == (2, 0, 0)
    x = solve_linear_system!(M, T[1.0, 3.0])
    return @test isapprox(x, T.(SOL); atol = 1.0e-6)
end

# Umfpack is a general LU solver: pass the full (symmetric) matrix.
function test_lu_solver(S, T)
    csc = sparse(Int32[1, 1, 2, 2], Int32[1, 2, 1, 2], T[1.0, 0.1, 0.1, 2.0], 2, 2)
    M = S(csc; opt = default_options(S))
    introduce(M)
    improve!(M)
    factorize!(M)
    x = solve_linear_system!(M, T[1.0, 3.0])
    return @test isapprox(x, T.(SOL); atol = 1.0e-6)
end

@testset "MadCoreSuiteSparse" begin
    @testset "CHOLMODSolver" begin
        test_symmetric_solver(CHOLMODSolver, Float64)
    end
    @testset "UmfpackSolver" begin
        test_lu_solver(UmfpackSolver, Float64)
    end
end
