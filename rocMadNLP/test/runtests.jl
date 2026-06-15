using Test, MadNLP, MadCoreAMDGPU, rocMadNLP, MadNLPTests

const HAS_AMDGPU = try
    using AMDGPU
    true
catch err
    @info "AMDGPU could not be loaded" exception = err
    false
end

@testset "rocMadNLP" begin
    if HAS_AMDGPU && AMDGPU.functional()
        include("densekkt_rocm.jl")
        # CompactLBFGS in SparseCondensedKKTSystem is not yet supported (Issue #563);
        # disabled upstream as well.
        # include("sparsekkt_rocm.jl")
    else
        @info "AMDGPU not functional — skipping rocMadNLP GPU tests" HAS_AMDGPU
        @test_skip true
    end
end
