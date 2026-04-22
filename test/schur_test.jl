using Test
using LinearAlgebra
using MadNLP
using MadNLPTests

@testset "SchurComplementKKTSystem" begin

    @testset "Basic convergence — quadratic with coupling" begin
        # min sum_k (v_k - θ_k)^2 + (d - 1)^2
        # s.t. v_k + d = 0 for k = 1..ns
        ns, nv, nd, nc = 3, 1, 1, 1
        θ = [4.0, 6.0, 8.0]

        qp = build_twostage_qp(;
            ns, nv, nd, nc,
            hess_v = fill(2.0, nv, ns),
            hess_d = fill(2.0, nd),
            g_v = reshape(-2 .* θ, nv, ns),
            g_d = [-2.0],
            A_v = fill(1.0, nc, nv, ns),
            A_d = fill(1.0, nc, nd, ns),
            lcon = zeros(nc, ns), ucon = zeros(nc, ns),
            lvar_v = fill(-100.0, nv, ns), uvar_v = fill(100.0, nv, ns),
            lvar_d = fill(-100.0, nd), uvar_d = fill(100.0, nd),
        )

        result = madnlp(
            qp;
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCPUSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        # Analytic optimum: d* = (1 - sum θ) / (ns + 1), v_k* = -d*
        d_star = (1.0 - sum(θ)) / (ns + 1)
        @test isapprox(result.solution[end], d_star; atol = 1.0e-3)
        @test all(isapprox(result.solution[k], -d_star; atol = 1.0e-3) for k in 1:ns)
    end

    @testset "Match SparseKKT reference" begin
        # min sum_{k,j} (v_{k,j} - θ_{k,j})^2 + d^2
        # s.t. v_{k,1} + v_{k,2} + d = 0
        ns, nv, nd, nc = 2, 2, 1, 1
        θ = [1.0 3.0; 2.0 4.0]   # θ[j, k]

        function mk()
            build_twostage_qp(;
                ns, nv, nd, nc,
                hess_v = fill(2.0, nv, ns),
                hess_d = fill(2.0, nd),
                g_v = -2 .* θ,
                g_d = [0.0],
                A_v = fill(1.0, nc, nv, ns),
                A_d = fill(1.0, nc, nd, ns),
                lcon = zeros(nc, ns), ucon = zeros(nc, ns),
                lvar_v = fill(-50.0, nv, ns), uvar_v = fill(50.0, nv, ns),
                lvar_d = fill(-50.0, nd), uvar_d = fill(50.0, nd),
            )
        end

        ref = madnlp(mk(); linear_solver = LapackCPUSolver, print_level = MadNLP.ERROR)
        schur = madnlp(
            mk();
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCPUSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )

        @test ref.status == MadNLP.SOLVE_SUCCEEDED
        @test schur.status == MadNLP.SOLVE_SUCCEEDED
        @test isapprox(schur.objective, ref.objective; atol = 1.0e-6)
        @test isapprox(schur.solution, ref.solution; atol = 1.0e-4)
    end

    @testset "Multiple recourse vars and design vars" begin
        # min sum_{k,j} θ_k * v_{k,j}^2 + sum_j d_j^2
        # s.t. v_{k,1} + v_{k,2} + d_1 + d_2 = 1
        ns, nv, nd, nc = 2, 2, 2, 1
        θ = [1.0, 2.0]

        H_v = zeros(nv, ns)
        for k in 1:ns, j in 1:nv
            H_v[j, k] = 2θ[k]
        end

        qp = build_twostage_qp(;
            ns, nv, nd, nc,
            hess_v = H_v, hess_d = fill(2.0, nd),
            g_v = zeros(nv, ns), g_d = zeros(nd),
            A_v = fill(1.0, nc, nv, ns),
            A_d = fill(1.0, nc, nd, ns),
            lcon = ones(nc, ns), ucon = ones(nc, ns),
            lvar_v = fill(-50.0, nv, ns), uvar_v = fill(50.0, nv, ns),
            lvar_d = fill(-50.0, nd), uvar_d = fill(50.0, nd),
        )

        result = madnlp(
            qp;
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCPUSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Known solution with inactive constraints" begin
        # min sum_k (v_k - θ_k)^2 + (d - 5)^2
        # s.t. -100 <= v_k + d <= 100  (inactive at optimum)
        # Solution: v_k* = θ_k, d* = 5
        ns, nv, nd, nc = 2, 1, 1, 1
        θ = [3.0, 7.0]

        qp = build_twostage_qp(;
            ns, nv, nd, nc,
            hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
            g_v = reshape(-2 .* θ, nv, ns), g_d = [-10.0],
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = fill(-100.0, nc, ns), ucon = fill(100.0, nc, ns),
            lvar_v = fill(-100.0, nv, ns), uvar_v = fill(100.0, nv, ns),
            lvar_d = fill(-100.0, nd), uvar_d = fill(100.0, nd),
        )

        result = madnlp(
            qp;
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCPUSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        @test isapprox(result.solution[1], 3.0; atol = 1.0e-3)
        @test isapprox(result.solution[2], 7.0; atol = 1.0e-3)
        @test isapprox(result.solution[3], 5.0; atol = 1.0e-3)
    end
end
