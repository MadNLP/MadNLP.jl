using Test, CUDA, AMDGPU, MadNLP, MadNLPGPU, MadNLPTests

@testset "MadNLPGPU test" begin
    include("madnlpgpu_test.jl")
    if CUDA.functional()
        include("densekkt_gpu.jl")
    end
end
