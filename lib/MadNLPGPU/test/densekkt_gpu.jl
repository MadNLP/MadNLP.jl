
using CUDA
using MadNLPTests

function _compare_gpu_with_cpu(KKTSystem, n, m, ind_fixed)

    opt_kkt = if (KKTSystem == MadNLP.DenseKKTSystem)
        MadNLP.DENSE_KKT_SYSTEM
    elseif (KKTSystem == MadNLP.DenseCondensedKKTSystem)
        MadNLP.DENSE_CONDENSED_KKT_SYSTEM
    end

    for (T,tol,atol) in [(Float32,1e-3,1e-1), (Float64,1e-8,1e-6)]
        madnlp_options = Dict{Symbol, Any}(
            :kkt_system=>opt_kkt,
            :linear_solver=>LapackGPUSolver,
            :print_level=>MadNLP.ERROR,
            :tol=>tol
        )

        nlp = MadNLPTests.DenseDummyQP{T}(; n=n, m=m, fixed_variables=ind_fixed)

        # Solve on CPU
        h_solver = MadNLP.MadNLPSolver(nlp; madnlp_options...)
        MadNLP.solve!(h_solver)

        # Solve on GPU
        d_solver = MadNLPGPU.CuMadNLPSolver(nlp; madnlp_options...)
        MadNLP.solve!(d_solver)

        @test isa(d_solver.kkt, KKTSystem{T, CuVector{T}, CuMatrix{T}})
        # # Check that both results match exactly
        @test h_solver.cnt.k == d_solver.cnt.k
        @test h_solver.obj_val ≈ d_solver.obj_val atol=atol
        @test h_solver.x ≈ d_solver.x atol=atol
        @test h_solver.y ≈ d_solver.y atol=atol
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
