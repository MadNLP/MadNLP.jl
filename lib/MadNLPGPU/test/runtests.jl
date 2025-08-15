using Test, CUDA, MadNLP, MadNLPGPU, MadNLPTests

@testset "MadNLPGPU test" begin
    if CUDA.functional()
        include("madnlpgpu_test.jl")
        include("densekkt_gpu.jl")
    end
end
