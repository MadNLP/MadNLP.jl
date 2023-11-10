using Test, CUDA, MadNLP, MadNLPGPU, MadNLPTests


# Test DenseKKTSystem on GPU

@testset "MadNLPGPU test" begin
    include("madnlpgpu_test.jl")
    include("densekkt_gpu.jl")
end
