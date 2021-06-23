using Test, MadNLP, JuMP, Plasmo
using MadNLPMumps, MadNLPHSL, MadNLPPardiso, MadNLPGraphs, MadNLPGPU, MadNLPIterative
import MathOptInterface
import AmplNLReader: AmplModel
import SparseArrays: sparse
import LinearAlgebra: norm

function solcmp(x,sol;atol=1e-4,rtol=1e-4)
    aerr = norm(x-sol,Inf)
    rerr = aerr/norm(sol,Inf)
    return (aerr < atol || rerr < rtol)
end

@testset "MadNLP test" begin
    @testset "Matrix tools" begin
        include("matrix_test.jl")
    end

    @testset "MOI interface" begin
        include("MOI_interface_test.jl")
    end

    @testset "NLPModels interface" begin
        include("nlpmodels_test.jl")
    end

    @testset "Plasmo interface" begin # this also serves as decomposition solver test
        include("plasmo_test.jl")
    end

    @testset "MadNLP test" begin 
        include("nlp_test.jl")
    end
end # @testset
