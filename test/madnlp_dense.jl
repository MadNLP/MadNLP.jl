using Test
import MadNLP: jac_structure!, hess_structure!, obj, grad!, cons!, jac_coord!, hess_coord!, jac_dense!, hess_dense!
using NLPModels
using LinearAlgebra
using MadNLPTests
using SparseArrays
using Random

@testset "MadNLP: dense API" begin
    n = 10
    @testset "Unconstrained" begin
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
            :linear_solver=>MadNLPLapackCPU,
        )
        m = 0
        nlp = MadNLPTests.DenseDummyQP(; n=n, m=m)
        ipd = MadNLP.InteriorPointSolver(nlp, option_dict=dense_options)

        kkt = ipd.kkt
        @test isa(kkt, MadNLP.DenseKKTSystem)
        @test isempty(kkt.jac)
        @test kkt.hess === kkt.aug_com
        @test ipd.linear_solver.dense === kkt.aug_com
        @test size(kkt.hess) == (n, n)
        @test length(kkt.pr_diag) == n
        @test length(kkt.du_diag) == m

        # Test that using a sparse solver is forbidden in dense mode
        dense_options_error = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
            :linear_solver=>MadNLPUmfpack,
        )
        @test_throws Exception MadNLP.InteriorPointSolver(nlp, dense_options_error)
    end
    @testset "Constrained" begin
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
            :linear_solver=>MadNLPLapackCPU,
        )
        m = 5
        nlp = MadNLPTests.DenseDummyQP(; n=n, m=m)
        ipd = MadNLP.InteriorPointSolver(nlp, option_dict=dense_options)
        ns = length(ipd.ind_ineq)

        kkt = ipd.kkt
        @test isa(kkt, MadNLP.DenseKKTSystem)
        @test size(kkt.jac) == (m, n + ns)
        @test ipd.linear_solver.dense === kkt.aug_com
        @test size(kkt.hess) == (n, n)
        @test length(kkt.pr_diag) == n + ns
        @test length(kkt.du_diag) == m
    end
end


function _compare_dense_with_sparse(n, m, ind_fixed)
    sparse_options = Dict{Symbol, Any}(
        :kkt_system=>MadNLP.SPARSE_KKT_SYSTEM,
        :linear_solver=>MadNLPLapackCPU,
        :print_level=>MadNLP.ERROR,
    )
    dense_options = Dict{Symbol, Any}(
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :linear_solver=>MadNLPLapackCPU,
        :print_level=>MadNLP.ERROR,
    )

    nlp = MadNLPTests.DenseDummyQP(; n=n, m=m, fixed_variables=ind_fixed)

    ips = MadNLP.InteriorPointSolver(nlp, option_dict=sparse_options)
    ipd = MadNLP.InteriorPointSolver(nlp, option_dict=dense_options)

    MadNLP.optimize!(ips)
    MadNLP.optimize!(ipd)

    # Check that dense formulation matches exactly sparse formulation
    @test ips.cnt.k == ipd.cnt.k
    @test ips.obj_val ≈ ipd.obj_val atol=1e-10
    @test ips.x ≈ ipd.x atol=1e-10
    @test ips.l ≈ ipd.l atol=1e-10
    @test ips.kkt.jac_com == ipd.kkt.jac
    @test Symmetric(ips.kkt.aug_com, :L) ≈ ipd.kkt.aug_com atol=1e-10
end

@testset "MadNLP: dense versus sparse" begin
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        _compare_dense_with_sparse(n, m, Int[])
    end
    @testset "Fixed variables" begin
        n, m = 10, 5
        _compare_dense_with_sparse(10, 5, Int[1, 2])
    end
end

