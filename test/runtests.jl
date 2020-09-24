using Test, MadNLP
using MathOptInterface, SparseArrays, LinearAlgebra, Plasmo, JuMP 

function solcmp(x,sol;atol=1e-4,rtol=1e-4)
    aerr = norm(x-sol,Inf)
    rerr = aerr/norm(sol,Inf)
    return (aerr < atol && rerr < rtol)
end

@testset "MadNLP test" begin
    @testset "Matrix tools" begin
        include("matrix_test.jl")
    end

    @testset "MOI interface" begin
        include("MOI_interface_test.jl")
    end

    @testset "Plasmo interface" begin # this also serves as decomposition solver test
        include("plasmo_test.jl")
    end

    @testset "NLP Algorithm" begin # this also serves as decomposition solver test
        include("nlp_test.jl")
    end
end # @testset
