using Test
using CUDA
using MadNLP
using MadNLPGPU
using ExaModels

# Helper to build a two-stage model using TwoStageExaCore + EachScenario
function build_schur_model(;ns, nv, nd, θ_vals, build_fn, vlvar=-100.0, vuvar=100.0, dlvar=-100.0, duvar=100.0, backend=nothing)
    core = TwoStageExaCore(ns; backend=backend)
    v = variable(core, nv, EachScenario(); lvar=vlvar, uvar=vuvar)
    d = variable(core, nd; lvar=dlvar, uvar=duvar)
    θ = parameter(core, θ_vals)
    build_fn(core, d, v, θ, ns, nv)
    return ExaModel(core)
end

@testset "GPUSchurComplementKKTSystem" begin

    @testset "Basic convergence — quadratic with coupling" begin
        model = build_schur_model(
            ns=3, nv=1, nd=1,
            θ_vals=[4.0, 6.0, 8.0],
            backend=CUDABackend(),
            build_fn=(c, d, v, θ, ns, nv) -> begin
                objective(c, (v[i] - θ[i])^2 for i in 1:ns)
                objective(c, (d[1] - 1.0)^2 for _ in 1:1)
                constraint(c, (v[i] + d[1] for i in 1:ns), EachScenario(); lcon=0.0)
            end,
        )

        result = madnlp(model;
            kkt_system=SchurComplementKKTSystem,
            linear_solver=LapackCUDASolver,
            print_level=MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Match CPU reference" begin
        build_fn = (c, d, v, θ, ns, nv) -> begin
            obj_data = [(i, (i-1)*nv+j, (i-1)*2+j) for i in 1:ns for j in 1:nv]
            objective(c, (v[vi] - θ[θi])^2 for (i, vi, θi) in obj_data)
            objective(c, d[1]^2 for _ in 1:1)
            con_data = [(i, (i-1)*nv+1, (i-1)*nv+2) for i in 1:ns]
            constraint(c, (v[v1] + v[v2] + d[1] for (i, v1, v2) in con_data), EachScenario(); lcon=0.0)
        end

        model_cpu = build_schur_model(
            ns=2, nv=2, nd=1,
            θ_vals=[1.0, 2.0, 3.0, 4.0],
            vlvar=-50.0, vuvar=50.0, dlvar=-50.0, duvar=50.0,
            build_fn=build_fn,
        )
        ref = madnlp(model_cpu;
            kkt_system=SchurComplementKKTSystem,
            linear_solver=LapackCPUSolver,
            print_level=MadNLP.ERROR,
        )

        model_gpu = build_schur_model(
            ns=2, nv=2, nd=1,
            θ_vals=[1.0, 2.0, 3.0, 4.0],
            vlvar=-50.0, vuvar=50.0, dlvar=-50.0, duvar=50.0,
            backend=CUDABackend(),
            build_fn=build_fn,
        )
        gpu_result = madnlp(model_gpu;
            kkt_system=SchurComplementKKTSystem,
            linear_solver=LapackCUDASolver,
            print_level=MadNLP.ERROR,
        )

        @test ref.status == MadNLP.SOLVE_SUCCEEDED
        @test gpu_result.status == MadNLP.SOLVE_SUCCEEDED
        @test isapprox(gpu_result.objective, ref.objective; atol=1e-6)
        @test isapprox(Array(gpu_result.solution), Array(ref.solution); atol=1e-4)
    end

    @testset "Multiple recourse vars and design vars" begin
        model = build_schur_model(
            ns=2, nv=2, nd=2,
            θ_vals=[1.0, 2.0],
            vlvar=-50.0, vuvar=50.0, dlvar=-50.0, duvar=50.0,
            backend=CUDABackend(),
            build_fn=(c, d, v, θ, ns, nv) -> begin
                obj_data = [(i, (i-1)*nv+j, i) for i in 1:ns for j in 1:nv]
                objective(c, θ[θi] * v[vi]^2 for (i, vi, θi) in obj_data)
                objective(c, d[j]^2 for j in 1:2)
                con_data = [(i, (i-1)*nv+1, (i-1)*nv+2) for i in 1:ns]
                constraint(c, (v[v1] + v[v2] + d[1] + d[2] for (i, v1, v2) in con_data), EachScenario(); lcon=1.0, ucon=1.0)
            end,
        )

        result = madnlp(model;
            kkt_system=SchurComplementKKTSystem,
            linear_solver=LapackCUDASolver,
            print_level=MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Known solution with inactive constraints" begin
        model = build_schur_model(
            ns=2, nv=1, nd=1,
            θ_vals=[3.0, 7.0],
            backend=CUDABackend(),
            build_fn=(c, d, v, θ, ns, nv) -> begin
                objective(c, (v[i] - θ[i])^2 for i in 1:ns)
                objective(c, (d[1] - 5.0)^2 for _ in 1:1)
                constraint(c, (v[i] + d[1] for i in 1:ns), EachScenario(); lcon=-100.0, ucon=100.0)
            end,
        )

        result = madnlp(model;
            kkt_system=SchurComplementKKTSystem,
            linear_solver=LapackCUDASolver,
            print_level=MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        sol = Array(result.solution)
        @test isapprox(sol[1], 3.0; atol=1e-3)
        @test isapprox(sol[2], 7.0; atol=1e-3)
        @test isapprox(sol[3], 5.0; atol=1e-3)
    end
end
