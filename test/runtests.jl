using MadNLP
using Test

function solcmp(x,sol,atol,rtol)
    aerr = norm(x-sol,Inf)
    rerr = aerr/norm(sol,Inf)
    return (aerr < atol && rerr < rtol)
end

@testset "MadNLP test" begin
    @testset "BLAS/Lapack" begin
        include("blas_lapack_test.jl")
    end

    @testset "Linear solvers" begin
        include("linear_solver_test.jl")
    end

    @testset "MOI interface" begin
        include("MOI_interface_test.jl")
    end

    @testset "CUTEst/NLPModels interfcae" begin # this also serves as interior point algorithm test
        include("CUTEst_test.jl")
    end

    @testset "Plasmo interface" begin # this also serves as decomposition solver test
        include("plasmo_test.jl")
    end
end # @testset
