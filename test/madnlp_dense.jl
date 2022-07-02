using Test
import MadNLP: jac_structure!, hess_structure!, obj, grad!, cons!, jac_coord!, hess_coord!, jac_dense!, hess_dense!
using NLPModels
using LinearAlgebra
using MadNLPTests
using SparseArrays
using Random

function _compare_dense_with_sparse(kkt_system, n, m, ind_fixed, ind_eq)

    for (T,tol,atol) in [(Float32,1e-3,1e-1), (Float64,1e-8,1e-6)]
        sparse_options = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.SPARSE_KKT_SYSTEM,
            :linear_solver=>MadNLP.LapackCPUSolver,
            :print_level=>MadNLP.ERROR,
        )
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>kkt_system,
            :linear_solver=>MadNLP.LapackCPUSolver,
            :print_level=>MadNLP.ERROR,
        )
        
        nlp = MadNLPTests.DenseDummyQP{T}(; n=n, m=m, fixed_variables=ind_fixed, equality_cons=ind_eq)

        ips = MadNLP.InteriorPointSolver(nlp, option_dict=sparse_options)
        ipd = MadNLP.InteriorPointSolver(nlp, option_dict=dense_options)

        MadNLP.optimize!(ips)
        MadNLP.optimize!(ipd)

        # Check that dense formulation matches exactly sparse formulation
        @test ips.cnt.k == ipd.cnt.k
        @test ips.obj_val ≈ ipd.obj_val atol=atol
        @test ips.x ≈ ipd.x atol=atol
        @test ips.l ≈ ipd.l atol=atol
        @test ips.kkt.jac_com[:, 1:n] == ipd.kkt.jac
        if isa(ipd.kkt, MadNLP.AbstractReducedKKTSystem)
            @test Symmetric(ips.kkt.aug_com, :L) ≈ ipd.kkt.aug_com atol=atol
        end
    end
end

@testset "MadNLP: API $(kkt_type)" for (kkt_type, kkt_options) in [
        (MadNLP.DenseKKTSystem, MadNLP.DENSE_KKT_SYSTEM),
        (MadNLP.DenseCondensedKKTSystem, MadNLP.DENSE_CONDENSED_KKT_SYSTEM),
    ]

    n = 10 # number of variables
    @testset "Unconstrained" begin
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>kkt_options,
            :linear_solver=>MadNLP.LapackCPUSolver,
        )
        m = 0
        nlp = MadNLPTests.DenseDummyQP(; n=n, m=m)
        ipd = MadNLP.InteriorPointSolver(nlp, option_dict=dense_options)

        kkt = ipd.kkt
        @test isa(kkt, kkt_type)
        @test isempty(kkt.jac)
        # Special test for DenseKKTSystem
        if kkt_type <: MadNLP.DenseKKTSystem
            @test kkt.hess === kkt.aug_com
        end
        @test ipd.linear_solver.dense === kkt.aug_com
        @test size(kkt.hess) == (n, n)
        @test length(kkt.pr_diag) == n
        @test length(kkt.du_diag) == m

        # Test that using a sparse solver is forbidden in dense mode
        dense_options_error = Dict{Symbol, Any}(
            :kkt_system=>kkt_options,
            :linear_solver=>MadNLP.UmfpackSolver,
        )
        @test_throws Exception MadNLP.InteriorPointSolver(nlp, dense_options_error)
    end
    @testset "Constrained" begin
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
            :linear_solver=>MadNLP.LapackCPUSolver,
        )
        m = 5
        nlp = MadNLPTests.DenseDummyQP(; n=n, m=m)
        ipd = MadNLP.InteriorPointSolver(nlp, option_dict=dense_options)
        ns = length(ipd.ind_ineq)

        kkt = ipd.kkt
        @test isa(kkt, MadNLP.DenseKKTSystem)
        @test size(kkt.jac) == (m, n)
        @test ipd.linear_solver.dense === kkt.aug_com
        @test size(kkt.hess) == (n, n)
        @test length(kkt.pr_diag) == n + ns
        @test length(kkt.du_diag) == m
    end
end


@testset "MadNLP: option kkt_system=$(kkt_system)" for kkt_system in [MadNLP.DENSE_KKT_SYSTEM, MadNLP.DENSE_CONDENSED_KKT_SYSTEM]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        _compare_dense_with_sparse(kkt_system, n, m, Int[], Int[])
    end
    @testset "Equality constraints" begin
        n, m = 20, 15
        _compare_dense_with_sparse(kkt_system, n, m, Int[], Int[1, 8]) # test with non-trivial equality constraints
    end
    @testset "Fixed variables" begin
        n, m = 10, 5
        _compare_dense_with_sparse(kkt_system, n, m, Int[1, 2], Int[])
    end
end

@testset "MadNLP: restart (PR #113)" begin
    n, m = 10, 5
    nlp = MadNLPTests.DenseDummyQP(; n=n, m=m)
    sparse_options = Dict{Symbol, Any}(
        :kkt_system=>MadNLP.SPARSE_KKT_SYSTEM,
        :linear_solver=>MadNLP.LapackCPUSolver,
        :print_level=>MadNLP.ERROR,
    )
    ips = MadNLP.InteriorPointSolver(nlp, option_dict=sparse_options)
    MadNLP.optimize!(ips)
    # Restart (should hit MadNLP.reinitialize function)
    res = MadNLP.optimize!(ips)
    @test ips.status == MadNLP.SOLVE_SUCCEEDED
end
