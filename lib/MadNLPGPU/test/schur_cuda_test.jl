using Test
using CUDA
using MadNLP
using MadNLPGPU
using ExaModels

@testset "GPUSchurComplementKKTSystem" begin

    @testset "Basic convergence — quadratic with coupling" begin
        ns, nv, nd = 3, 1, 1
        nc = 1
        θ_sets = [[4.0], [6.0], [8.0]]

        tsm = TwoStageExaModel(nd, nv, ns, θ_sets;
            v_lvar = -100.0, v_uvar = 100.0,
            d_lvar = -100.0, d_uvar = 100.0,
            backend = CUDABackend(),
        ) do c, d, v, θ, ns, nv, nθ
            obj_data = [(i, (i-1)*nv+1, (i-1)*nθ+1) for i in 1:ns]
            objective(c, (v[vi] - θ[θi])^2 for (i, vi, θi) in obj_data)
            objective(c, (d[1] - 1.0)^2 for _ in 1:1)
            con_data = [(i, (i-1)*nv+1) for i in 1:ns]
            constraint(c, v[vi] + d[1] for (i, vi) in con_data; lcon = 0.0)
        end

        result = madnlp(tsm.model;
            kkt_system=SchurComplementKKTSystem,
            linear_solver=LapackCUDASolver,
            kkt_options=Dict{Symbol,Any}(:schur_ns=>ns, :schur_nv=>nv, :schur_nd=>nd, :schur_nc=>nc),
            print_level=MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Match CPU reference" begin
        ns, nv, nd = 2, 2, 1
        nc = 1
        θ_sets = [[1.0, 2.0], [3.0, 4.0]]

        function build_model_cpu()
            TwoStageExaModel(nd, nv, ns, θ_sets;
                v_lvar = -50.0, v_uvar = 50.0,
                d_lvar = -50.0, d_uvar = 50.0,
            ) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, (i-1)*nv+j, (i-1)*2+j) for i in 1:ns for j in 1:nv]
                objective(c, (v[vi] - θ[θi])^2 for (i, vi, θi) in obj_data)
                objective(c, d[1]^2 for _ in 1:1)
                con_data = [(i, (i-1)*nv+1, (i-1)*nv+2) for i in 1:ns]
                constraint(c, v[v1] + v[v2] + d[1] for (i, v1, v2) in con_data; lcon = 0.0)
            end
        end

        function build_model_gpu()
            TwoStageExaModel(nd, nv, ns, θ_sets;
                v_lvar = -50.0, v_uvar = 50.0,
                d_lvar = -50.0, d_uvar = 50.0,
                backend = CUDABackend(),
            ) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, (i-1)*nv+j, (i-1)*2+j) for i in 1:ns for j in 1:nv]
                objective(c, (v[vi] - θ[θi])^2 for (i, vi, θi) in obj_data)
                objective(c, d[1]^2 for _ in 1:1)
                con_data = [(i, (i-1)*nv+1, (i-1)*nv+2) for i in 1:ns]
                constraint(c, v[v1] + v[v2] + d[1] for (i, v1, v2) in con_data; lcon = 0.0)
            end
        end

        # CPU reference
        tsm_cpu = build_model_cpu()
        ref = madnlp(tsm_cpu.model;
            kkt_system=SchurComplementKKTSystem,
            linear_solver=LapackCPUSolver,
            kkt_options=Dict{Symbol,Any}(:schur_ns=>ns, :schur_nv=>nv, :schur_nd=>nd, :schur_nc=>nc),
            print_level=MadNLP.ERROR,
        )

        # GPU
        tsm_gpu = build_model_gpu()
        gpu_result = madnlp(tsm_gpu.model;
            kkt_system=SchurComplementKKTSystem,
            linear_solver=LapackCUDASolver,
            kkt_options=Dict{Symbol,Any}(:schur_ns=>ns, :schur_nv=>nv, :schur_nd=>nd, :schur_nc=>nc),
            print_level=MadNLP.ERROR,
        )

        @test ref.status == MadNLP.SOLVE_SUCCEEDED
        @test gpu_result.status == MadNLP.SOLVE_SUCCEEDED
        @test isapprox(gpu_result.objective, ref.objective; atol=1e-6)
        @test isapprox(Array(gpu_result.solution), Array(ref.solution); atol=1e-4)
    end

    @testset "Multiple recourse vars and design vars" begin
        ns, nv, nd = 2, 2, 2
        nc = 1
        θ_sets = [[1.0], [2.0]]

        tsm = TwoStageExaModel(nd, nv, ns, θ_sets;
            v_lvar = -50.0, v_uvar = 50.0,
            d_lvar = -50.0, d_uvar = 50.0,
            backend = CUDABackend(),
        ) do c, d, v, θ, ns, nv, nθ
            obj_data = [(i, (i-1)*nv+j, (i-1)*nθ+1) for i in 1:ns for j in 1:nv]
            objective(c, θ[θi] * v[vi]^2 for (i, vi, θi) in obj_data)
            objective(c, d[j]^2 for j in 1:nd)
            con_data = [(i, (i-1)*nv+1, (i-1)*nv+2) for i in 1:ns]
            constraint(c, v[v1] + v[v2] + d[1] + d[2] for (i, v1, v2) in con_data; lcon = 1.0, ucon = 1.0)
        end

        result = madnlp(tsm.model;
            kkt_system=SchurComplementKKTSystem,
            linear_solver=LapackCUDASolver,
            kkt_options=Dict{Symbol,Any}(:schur_ns=>ns, :schur_nv=>nv, :schur_nd=>nd, :schur_nc=>nc),
            print_level=MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Known solution with inactive constraints" begin
        ns, nv, nd = 2, 1, 1
        nc = 1
        θ_sets = [[3.0], [7.0]]

        tsm = TwoStageExaModel(nd, nv, ns, θ_sets;
            v_lvar = -100.0, v_uvar = 100.0,
            d_lvar = -100.0, d_uvar = 100.0,
            backend = CUDABackend(),
        ) do c, d, v, θ, ns, nv, nθ
            obj_data = [(i, (i-1)*nv+1, (i-1)*nθ+1) for i in 1:ns]
            objective(c, (v[vi] - θ[θi])^2 for (i, vi, θi) in obj_data)
            objective(c, (d[1] - 5.0)^2 for _ in 1:1)
            con_data = [(i, (i-1)*nv+1) for i in 1:ns]
            constraint(c, v[vi] + d[1] for (i, vi) in con_data; lcon = -100.0, ucon = 100.0)
        end

        result = madnlp(tsm.model;
            kkt_system=SchurComplementKKTSystem,
            linear_solver=LapackCUDASolver,
            kkt_options=Dict{Symbol,Any}(:schur_ns=>ns, :schur_nv=>nv, :schur_nd=>nd, :schur_nc=>nc),
            print_level=MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        sol = Array(result.solution)
        @test isapprox(sol[1], 3.0; atol=1e-3)
        @test isapprox(sol[2], 7.0; atol=1e-3)
        @test isapprox(sol[3], 5.0; atol=1e-3)
    end
end
