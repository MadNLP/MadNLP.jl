using Test, CUDA, AMDGPU, oneAPI, MadNLP, MadNLPGPU, MadNLPTests

@testset "MadNLPGPU test" begin
    include("madnlpgpu_test.jl")
    if CUDA.functional()
        include("densekkt_cuda.jl")
    end
    if AMDGPU.functional()
        include("densekkt_rocm.jl")
    end
    # if oneAPI.functional()
    #     include("densekkt_oneapi.jl")
    # end
end
