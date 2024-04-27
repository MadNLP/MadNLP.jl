@testset "MadNLP: $QN + $KKT" for QN in [
    MadNLP.BFGS,
    MadNLP.DampedBFGS,
], KKT in [
    MadNLP.DenseKKTSystem,
    MadNLP.DenseCondensedKKTSystem,
]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        nlp = MadNLPTests.DenseDummyQP(zeros(Float64, n); m=m)
        solver_exact = MadNLP.MadNLPSolver(
            nlp;
            print_level=MadNLP.ERROR,
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=LapackCPUSolver,
        )
        results_ref = MadNLP.solve!(solver_exact)

        solver_qn = MadNLP.MadNLPSolver(
            nlp;
            print_level=MadNLP.ERROR,
            kkt_system=KKT,
            hessian_approximation=QN,
            linear_solver=LapackCPUSolver,
        )
        results_qn = MadNLP.solve!(solver_qn)

        @test results_qn.status == MadNLP.SOLVE_SUCCEEDED
        @test results_qn.objective ≈ results_ref.objective atol=1e-6
        @test results_qn.solution ≈ results_ref.solution atol=1e-6
        @test solver_qn.cnt.lag_hess_cnt == 0
        @test solver_exact.y ≈ solver_qn.y atol=1e-4
    end
end

@testset "MadNLP: LBFGS" begin
    @testset "HS15" begin
        nlp = MadNLPTests.HS15NoHessianModel()
        solver_qn = MadNLP.MadNLPSolver(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = MadNLP.SparseKKTSystem,
            hessian_approximation=MadNLP.CompactLBFGS,
            print_level=MadNLP.ERROR,
        )
        results_qn = MadNLP.solve!(solver_qn)
        @test results_qn.status == MadNLP.SOLVE_SUCCEEDED

    end
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        x0 = zeros(Float64,n)
        nlp = MadNLPTests.DenseDummyQP(x0; m=m)
        # Reference solve with exact Hessian
        solver_exact = MadNLP.MadNLPSolver(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = MadNLP.SparseKKTSystem,
            print_level=MadNLP.ERROR,
        )
        results_ref = MadNLP.solve!(solver_exact)

        # LBFGS solve
        solver_qn = MadNLP.MadNLPSolver(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = MadNLP.SparseKKTSystem,
            hessian_approximation=MadNLP.CompactLBFGS,
            print_level=MadNLP.ERROR,
        )
        results_qn = MadNLP.solve!(solver_qn)

        @test results_qn.status == MadNLP.SOLVE_SUCCEEDED
        @test results_qn.objective ≈ results_ref.objective atol=1e-6
        @test results_qn.solution ≈ results_ref.solution atol=1e-6
        @test solver_qn.cnt.lag_hess_cnt == 0
        @test solver_exact.y ≈ solver_qn.y atol=1e-4

        # Test accuracy of KKT solver with LBFGS
        b, x, w = solver_qn.p, solver_qn.d, solver_qn._w4
        fill!(b.values, 1.0)
        MadNLP.solve_refine_wrapper!(x, solver_qn, b, w)
        mul!(w, solver_qn.kkt, x)
        @assert norm(w.values .- b.values, Inf) <= 1e-6
    end
end

