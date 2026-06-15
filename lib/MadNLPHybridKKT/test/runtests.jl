using Test
using MadNLPHybridKKT

@testset "MadNLPHybridKKT" begin
    # Structural / load test: the package builds CUDA-free on MadCore + MadNLPCore.
    @test isdefined(MadNLPHybridKKT, :HybridCondensedKKTSystem)
    # NOTE: the full HyKKT GPU solve tests (elec_model on CUDA, from the original
    # HybridKKT.jl suite) require a GPU + the CUDA matrix glue and are a follow-up.
end
