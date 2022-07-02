
using CUDA
using MadNLPTests

function _compare_gpu_with_cpu(KKTSystem, n, m, ind_fixed)

    opt_kkt = if (KKTSystem == MadNLP.DenseKKTSystem)
        MadNLP.DENSE_KKT_SYSTEM
    elseif (KKTSystem == MadNLP.DenseCondensedKKTSystem)
        MadNLP.DENSE_CONDENSED_KKT_SYSTEM
    end
    # Define options
    madnlp_options = Dict{Symbol, Any}(
        :kkt_system=>opt_kkt,
        :linear_solver=>LapackGPUSolver,
        :print_level=>MadNLP.ERROR,
    )

    for (T,tol,atol) in [(Float32,1e-3,1e-1), (Float64,1e-8,1e-6)]
        madnlp_options = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
            :linear_solver=>MadNLPLapackGPU,
            :print_level=>MadNLP.ERROR,
            :tol=>tol
        )
        # Solve on CPU
        h_ips = MadNLP.InteriorPointSolver(nlp; option_dict=copy(madnlp_options))
        MadNLP.optimize!(h_ips)

        # Solve on GPU
        d_ips = MadNLPGPU.CuInteriorPointSolver(nlp; option_dict=copy(madnlp_options))
        MadNLP.optimize!(d_ips)

        T = Float64
        VT = CuVector{T}
        MT = CuMatrix{T}
        @test isa(d_ips.kkt, KKTSystem{T, VT, MT})
        # # Check that both results match exactly
        @test h_ips.cnt.k == d_ips.cnt.k
        @test h_ips.obj_val ≈ d_ips.obj_val atol=1e-10
        @test h_ips.x ≈ d_ips.x atol=1e-10
        @test h_ips.l ≈ d_ips.l atol=1e-10
    end
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
