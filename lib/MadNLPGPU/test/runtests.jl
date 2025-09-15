using Test, CUDA, AMDGPU, oneAPI, MadNLP, MadNLPGPU, MadNLPTests

@testset "MadNLPGPU test" begin
    include("madnlpgpu_test.jl")
    if CUDA.functional()
        include("densekkt_cuda.jl")
        # Need to add support for CompactLBFGS in SparseCondensedKKTSystem (Issue #563)
        # include("sparsekkt_cuda.jl")
    end
    if AMDGPU.functional()
        include("densekkt_rocm.jl")
        # Need to add support for CompactLBFGS in SparseCondensedKKTSystem (Issue #563)
        # include("sparsekkt_rocm.jl")
    end
end
