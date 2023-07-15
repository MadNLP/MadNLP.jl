@testset "MadNLP: $QN + $KKT" for QN in [
    MadNLP.DENSE_BFGS,
    MadNLP.DENSE_DAMPED_BFGS,
], KKT in [
    MadNLP.DenseKKTSystem,
    MadNLP.DenseCondensedKKTSystem,
]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        nlp = MadNLPTests.DenseDummyQP{Float64}(; n=n, m=m)
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
        # TODO: this test is currently breaking the CI, investigate why.
        # @test solver_exact.y ≈ solver_qn.y atol=1e-4
    end
end

@testset "MadNLP: LBFGS" begin
    @testset "HS15" begin
        nlp = MadNLPTests.HS15Model()
        solver_qn = MadNLP.MadNLPSolver(
            nlp;
            hessian_approximation=MadNLP.SPARSE_COMPACT_LBFGS,
            print_level=MadNLP.ERROR,
        )
        results_qn = MadNLP.solve!(solver_qn)
        @test results_qn.status == MadNLP.SOLVE_SUCCEEDED
    end
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        nlp = MadNLPTests.DenseDummyQP{Float64}(; )
        # Reference solve with exact Hessian
        solver_exact = MadNLP.MadNLPSolver(
            nlp;
            print_level=MadNLP.ERROR,
        )
        results_ref = MadNLP.solve!(solver_exact)

        # LBFGS solve
        solver_qn = MadNLP.MadNLPSolver(
            nlp;
            hessian_approximation=MadNLP.SPARSE_COMPACT_LBFGS,
            print_level=MadNLP.ERROR,
        )
        results_qn = MadNLP.solve!(solver_qn)
        @test results_qn.status == MadNLP.SOLVE_SUCCEEDED
        @test results_qn.objective ≈ results_ref.objective atol=1e-6
        @test results_qn.solution ≈ results_ref.solution atol=1e-6
        @test solver_qn.cnt.lag_hess_cnt == 0
        # TODO: this test is currently breaking the CI, investigate why.
        # @test solver_exact.y ≈ solver_qn.y atol=1e-4
    end
end

