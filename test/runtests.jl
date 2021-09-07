using Test, MadNLP, MadNLPTests, MINLPTests
import MathOptInterface
import AmplNLReader: AmplModel
import SparseArrays: sparse

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

    @testset "MadNLP test" begin
        include("madnlp_test.jl")
        include("madnlp_dense.jl")
    end

    @testset "MINLP test" begin
        include("minlp_test.jl")
    end
end # @testset
