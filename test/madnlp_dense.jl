using Test
import MadNLP: jac_structure!, hess_structure!, obj, grad!, cons!, jac_coord!, hess_coord!, jac_dense!, hess_dense!
using NLPModels
using LinearAlgebra
using SparseArrays
using Random

struct DenseDummyQP <: AbstractNLPModel{Float64,Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    P::Matrix{Float64} # primal hessian
    A::Matrix{Float64} # constraint jacobian
    q::Vector{Float64}
    hrows::Vector{Int}
    hcols::Vector{Int}
    jrows::Vector{Int}
    jcols::Vector{Int}
    counters::Counters
end


function jac_structure!(qp::DenseDummyQP,I, J)
    copyto!(I, qp.jrows)
    copyto!(J, qp.jcols)
end
function hess_structure!(qp::DenseDummyQP,I, J)
    copyto!(I, qp.hrows)
    copyto!(J, qp.hcols)
end

function obj(qp::DenseDummyQP,x)
    return 0.5 * dot(x, qp.P, x) + dot(qp.q, x)
end
function grad!(qp::DenseDummyQP,x,g)
    mul!(g, qp.P, x)
    g .+= qp.q
    return
end
function cons!(qp::DenseDummyQP,x,c)
    mul!(c, qp.A, x)
end
# Jacobian: sparse callback
function jac_coord!(qp::DenseDummyQP, x, J::AbstractVector)
    index = 1
    for (i, j) in zip(qp.jrows, qp.jcols)
        J[index] =  qp.A[i, j]
        index += 1
    end
end
# Jacobian: dense callback
jac_dense!(qp::DenseDummyQP, x, J::AbstractMatrix) = copyto!(J, qp.A)
# Hessian: sparse callback
function hess_coord!(qp::DenseDummyQP,x, l, hess::AbstractVector; obj_weight=1.)
    index = 1
    for i in 1:get_nvar(qp) , j in 1:i
        hess[index] = obj_weight * qp.P[j, i]
        index += 1
    end
end
# Hessian: dense callback
function hess_dense!(qp::DenseDummyQP, x, l,hess::AbstractMatrix; obj_weight=1.)
    hess .= obj_weight .* qp.P
end


function DenseDummyQP(; n=100, m=10, fixed_variables=Int[])
    if m >= n
        error("The number of constraints `m` should be less than the number of variable `n`.")
    end

    Random.seed!(1)

    # Build QP problem 0.5 * x' * P * x + q' * x
    P = randn(n , n)
    P += P' # P is symmetric
    P += 100.0 * I

    q = randn(n)

    # Build constraints gl <= Ax <= gu
    A = zeros(m, n)
    for j in 1:m
        A[j, j]  = 1.0
        A[j, j+1]  = -1.0
    end

    x0 = zeros(n)
    y0 = zeros(m)

    # Bound constraints
    xu =   ones(n)
    xl = - ones(n)
    gl = -ones(m)
    gu = ones(m)

    xl[fixed_variables] .= xu[fixed_variables]

    hrows = [i for i in 1:n for j in 1:i]
    hcols = [j for i in 1:n for j in 1:i]
    nnzh = div(n * (n + 1), 2)

    jrows = [j for i in 1:n for j in 1:m]
    jcols = [i for i in 1:n for j in 1:m]
    nnzj = n * m

    return DenseDummyQP(
        NLPModels.NLPModelMeta(
            n,
            ncon = m,
            nnzj = nnzj,
            nnzh = nnzh,
            x0 = x0,
            y0 = y0,
            lvar = xl,
            uvar = xu,
            lcon = gl,
            ucon = gu,
            minimize = true
        ),
        P,A,q,hrows,hcols,jrows,jcols,
        Counters()
    )
end

@testset "MadNLP: dense API" begin
    n = 10
    @testset "Unconstrained" begin
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
            :linear_solver=>MadNLPLapackCPU,
        )
        m = 0
        nlp = DenseDummyQP(; n=n, m=m)
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
        nlp = DenseDummyQP(; n=n, m=m)
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

    nlp = DenseDummyQP(; n=n, m=m, fixed_variables=ind_fixed)

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

