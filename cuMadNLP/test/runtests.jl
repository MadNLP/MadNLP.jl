using Test, MadNLP, MadCoreCUDA, MadCoreKernelAbstractions, cuMadNLP, MadNLPTests

const HAS_CUDA = try
    using CUDA
    using CUDSS
    true
catch err
    @info "CUDA/CUDSS could not be loaded" exception = err
    false
end

@testset "cuMadNLP" begin
    if HAS_CUDA && CUDA.functional()
        include("densekkt_cuda.jl")
        include("sparsekkt_cuda.jl")
        include("schur_cuda_test.jl")
        include("cuda_moi_test.jl")
    else
        @info "CUDA not functional — skipping cuMadNLP GPU tests" HAS_CUDA
        @test_skip true
    end
end
