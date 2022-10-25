
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
        results_cpu = MadNLP.solve!(h_solver)

        # Solve on GPU
        d_solver = MadNLPGPU.CuMadNLPSolver(nlp; madnlp_options...)
        results_gpu = MadNLP.solve!(d_solver)

        @test isa(d_solver.kkt, KKTSystem{T, CuVector{T}, CuMatrix{T}})
        # # Check that both results match exactly
        @test h_solver.cnt.k == d_solver.cnt.k
        @test results_cpu.objective ≈ results_gpu.objective
        @test results_cpu.solution ≈ results_gpu.solution atol=atol
        @test results_cpu.multipliers ≈ results_gpu.multipliers atol=atol
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

@testset "MadNLP: $QN + $KKT" for QN in [
    MadNLP.DENSE_BFGS,
    MadNLP.DENSE_DAMPED_BFGS,
], KKT in [
    MadNLP.DENSE_KKT_SYSTEM,
    MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        nlp = MadNLPTests.DenseDummyQP{Float64}(; n=n, m=m)
        solver_exact = MadNLP.MadNLPSolver(
            nlp;
            print_level=MadNLP.ERROR,
            kkt_system=KKT,
            linear_solver=LapackGPUSolver,
        )
        MadNLP.solve!(solver_exact)

        solver_qn = MadNLPGPU.CuMadNLPSolver(
            nlp;
            print_level=MadNLP.ERROR,
            kkt_system=KKT,
            hessian_approximation=QN,
            linear_solver=LapackGPUSolver,
        )
        MadNLP.solve!(solver_qn)

        @test solver_qn.status == MadNLP.SOLVE_SUCCEEDED
        @test solver_qn.cnt.lag_hess_cnt == 0
        @test solver_exact.obj_val ≈ solver_qn.obj_val atol=1e-6
        @test solver_exact.x ≈ solver_qn.x atol=1e-6
    end
end

