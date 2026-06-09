using Test, MadNLP, MadCoreCUDA, MadCoreKernelAbstractions, CuMadNLP, MadNLPTests

const HAS_CUDA = try
    using CUDA
    using CUDSS
    CUDA.functional()
catch
    false
end

@testset "CuMadNLP" begin
    if HAS_CUDA
        include("densekkt_cuda.jl")
        include("schur_cuda_test.jl")
    else
        @warn "CUDA not functional — skipping CuMadNLP GPU tests"
    end
end
