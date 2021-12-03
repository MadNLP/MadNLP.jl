
using CUDA
using MadNLPTests

function _compare_gpu_with_cpu(kkt_system, n, m, ind_fixed)
    madnlp_options = Dict{Symbol, Any}(
        :kkt_system=>MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
        :linear_solver=>MadNLPLapackGPU,
        :print_level=>MadNLP.ERROR,
    )

    nlp = MadNLPTests.DenseDummyQP(; n=n, m=m, fixed_variables=ind_fixed)

    h_ips = MadNLP.InteriorPointSolver(nlp; option_dict=copy(madnlp_options))
    MadNLP.optimize!(h_ips)


    # Init KKT on the GPU
    TKKTGPU = kkt_system{Float64, CuVector{Float64}, CuMatrix{Float64}}
    opt = MadNLP.Options(; madnlp_options...)
    # Instantiate Solver with KKT on the GPU
    d_ips = MadNLP.InteriorPointSolver{TKKTGPU}(nlp, opt; option_linear_solver=copy(madnlp_options))
    MadNLP.optimize!(d_ips)

    # # Check that both results match exactly
    @test h_ips.cnt.k == d_ips.cnt.k
    @test h_ips.obj_val ≈ d_ips.obj_val atol=1e-10
    @test h_ips.x ≈ d_ips.x atol=1e-10
    @test h_ips.l ≈ d_ips.l atol=1e-10
end

@testset "MadNLPGPU ($(kkt_system))" for kkt_system in [
        MadNLP.DenseKKTSystem,
        MadNLP.DenseCondensedKKTSystem,
    ]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        _compare_gpu_with_cpu(kkt_system, n, m, Int[])
    end
    @testset "Fixed variables" begin
        n, m = 20, 0 # warning: setting m >= 1 does not work in inertia free mode
        _compare_gpu_with_cpu(kkt_system, n, m, Int[1, 2])
    end
end
