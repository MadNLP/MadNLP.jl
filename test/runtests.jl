using Test, MadNLP, MadNLPTests, MINLPTests
using NLPModels
using Quadmath
import MathOptInterface
import SparseArrays: sparse

@testset "MadNLP test" begin
    @testset "Matrix tools" begin
        include("matrix_test.jl")
    end

    @testset "KKTSystem" begin
        include("kkt_test.jl")
    end

    @testset "Linear solvers" begin
        include("dummy_solver_test.jl")
    end

    @testset "MOI interface" begin
        include("MOI_interface_test.jl")
    end

    @testset "MadNLP test" begin
        include("madnlp_test.jl")
        include("madnlp_dense.jl")
        include("madnlp_quasi_newton.jl")
    end

    @testset "MINLP test" begin
        include("minlp_test.jl")
    end

    @testset "Schur complement" begin
        include("schur_test.jl")
    end

    # Solver-integration tests incorporated from the old per-solver lib packages.
    # (MUMPS is the default and is exercised throughout; Krylov was retired; the
    # GPU integration tests live in lib/cuMadNLP/test.)
    @testset "Pardiso" begin
        include("madnlp_pardiso_test.jl")
    end

    @testset "HSL" begin
        include("madnlp_hsl_test.jl")
    end
end # @testset
