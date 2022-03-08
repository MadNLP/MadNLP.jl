using Test, MadNLP, MadNLPTests, MINLPTests
import MathOptInterface
# import AmplNLReader: AmplModel
import SparseArrays: sparse

@testset "MadNLP test" begin
    @testset "Matrix tools" begin
        include("matrix_test.jl")
    end

    @testset "MOI interface" begin
        include("MOI_interface_test.jl")
    end

    # this is temporarily commented out due to the incompatibility between NLPModels v0.17.2 and AmplNLReader v0.11.0
    # @testset "NLPModels interface" begin
    #     include("nlpmodels_test.jl")
    # end

    @testset "MadNLP test" begin
        include("madnlp_test.jl")
        include("madnlp_dense.jl")
    end

    @testset "MINLP test" begin
        include("minlp_test.jl")
    end
end # @testset
