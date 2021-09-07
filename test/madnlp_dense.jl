using Test
using LinearAlgebra
using SparseArrays
using Random


function build_qp_test(; n=100, m=10, fixed_variables=Int[])
    if m >= n
        error("The number of constraints `m` should be less than the number of variable `n`.")
    end

    Random.seed!(1)
    # Build QP problem 0.5 * x' * P * x + q' * x
    P = randn(n , n)
    P += P' # P is symmetric
    P += 2.0 * I
    q = randn(n)

    # Build constraints gl <= Ax <= gu
    A = zeros(m, n)
    for j in 1:m
        A[j, j]  = 1.0
        A[j, j+1]  = -1.0
    end

    x0 = zeros(n)
    # Bound constraints
    xu =   ones(n)
    xl = - ones(n)
    gl = -ones(m)
    gu = ones(m)

    xl[fixed_variables] .= xu[fixed_variables]

    rows = [i for i in 1:n for j in 1:i]
    cols = [j for i in 1:n for j in 1:i]
    nnzh = div(n * (n + 1), 2)

    jrows = [j for i in 1:n for j in 1:m]
    jcols = [i for i in 1:n for j in 1:m]
    nnzj = n * m
    function jac_struct(I, J)
        copyto!(I, jrows)
        copyto!(J, jcols)
    end
    function hess_struct(I, J)
        copy!(I, rows)
        copy!(J, cols)
    end

    function eval_f(x)
        return 0.5 * dot(x, P, x) + dot(q, x)
    end
    function eval_g(g, x)
        mul!(g, P, x)
        g .+= q
        return
    end
    eval_cons(c, x) = mul!(c, A, x)
    # Jacobian: sparse callback
    function eval_jac(J::AbstractVector, x)
        index = 1
        for (i, j) in zip(jrows, jcols)
            J[index] =  A[i, j]
            index += 1
        end
    end
    # Jacobian: dense callback
    eval_jac(J::AbstractMatrix, x) = copyto!(J, A)
    # Hessian: sparse callback
    function eval_hess(hess::AbstractVector, x, l, sig)
        index = 1
        for i in 1:n , j in 1:i
            hess[index] = sig * P[j, i]
            index += 1
        end
    end
    # Hessian: dense callback
    function eval_hess(hess::AbstractMatrix, x, l, sig)
        copyto!(hess, sig .* P)
    end

    return MadNLP.NonlinearProgram(
        n, m, nnzh, nnzj,
        0.0, x0,
        zeros(m), zeros(m),
        zeros(n), zeros(n),
        xl, xu, gl, gu,
        eval_f,
        eval_g,
        eval_cons,
        eval_jac,
        eval_hess,
        hess_struct,
        jac_struct,
        MadNLP.INITIAL,
        Dict{Symbol,Any}(),
    )
end

function reset_nlp!(nlp::MadNLP.NonlinearProgram, x0, l0)
    copyto!(nlp.x, x0)
    copyto!(nlp.l, l0)
end

function test_qp(; n=10, m=0, fixed_variables=Int[],
    linear_solver=MadNLPLapackCPU,
    kkt_system=MadNLP.SPARSE_KKT_SYSTEM,
    maxit=100,
)
    madnlp_options = Dict{Symbol, Any}(
        :max_iter=>maxit,
        :kkt_system=>kkt_system,
        :linear_solver=>linear_solver,
        :print_level=>MadNLP.DEBUG,
    )
    nlp = build_qp_test(; n=n, m=m, fixed_variables=fixed_variables)
    ipp = MadNLP.Solver(nlp; option_dict=madnlp_options)
    MadNLP.optimize!(ipp)
    return ipp
end

@testset "MadNLP: dense API" begin
    n = 10
    @testset "Unconstrained" begin
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
            :linear_solver=>MadNLPLapackCPU,
        )
        m = 0
        nlp = build_qp_test(; n=n, m=m)
        ipd = MadNLP.Solver(nlp, option_dict=dense_options)

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
        @test_throws Exception MadNLP.Solver(nlp, dense_options_error)
    end
    @testset "Constrained" begin
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
            :linear_solver=>MadNLPLapackCPU,
        )
        m = 5
        nlp = build_qp_test(; n=n, m=m)
        ipd = MadNLP.Solver(nlp, option_dict=dense_options)
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

    nlp = build_qp_test(; n=n, m=m, fixed_variables=ind_fixed)
    x, l = copy(nlp.x), copy(nlp.l)

    ips = MadNLP.Solver(nlp, option_dict=sparse_options)
    MadNLP.optimize!(ips)

    # Reinit NonlinearProgram to avoid side effect
    nlp = build_qp_test(; n=n, m=m, fixed_variables=ind_fixed)
    ipd = MadNLP.Solver(nlp, option_dict=dense_options)
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

