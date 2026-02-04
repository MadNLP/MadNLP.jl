using Test, CUDA, MadNLP, MadNLPGPU, MadNLPTests

# Get backend from environment
const GPU_BACKEND = get(ENV, "MADNLP_GPU_BACKEND", "cuda")

# Conditionally load GPU backends
if GPU_BACKEND == "amdgpu"
    using AMDGPU
elseif GPU_BACKEND == "oneapi"
    using oneAPI
end

@testset "MadNLPGPU test" begin
    include("madnlpgpu_test.jl")
    if GPU_BACKEND == "cuda" && CUDA.functional()
        include("densekkt_cuda.jl")
        # Need to add support for CompactLBFGS in SparseCondensedKKTSystem (Issue #563)
        # include("sparsekkt_cuda.jl")
    end
    if GPU_BACKEND == "amdgpu" && AMDGPU.functional()
        include("densekkt_rocm.jl")
        # Need to add support for CompactLBFGS in SparseCondensedKKTSystem (Issue #563)
        # include("sparsekkt_rocm.jl")
    end
    if GPU_BACKEND == "oneapi" && oneAPI.functional()
        include("densekkt_oneapi.jl")
    end
end
