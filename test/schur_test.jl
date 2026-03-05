using Test
using LinearAlgebra
using MadNLP
import MadNLP.NLPModels

# Hand-rolled NLPModel for two-stage stochastic QPs used to exercise
# SchurComplementKKTSystem without pulling in ExaModels.
#
# Variable layout:   x = [v_{1,1},...,v_{1,nv}, v_{2,1},...,v_{ns,nv}, d_1,...,d_nd]
# Constraint layout: c = [c_{1,1},...,c_{1,nc}, c_{2,1},...,c_{ns,nc}]
# Hessian:  diagonal (each var has its own quadratic coef).
# Jacobian: per-scenario block-coupled (each scenario row touches its own v's + the shared d's).

struct TwoStageQP{T, VT <: AbstractVector{T}, MT <: AbstractMatrix{T}, VI <: AbstractVector{Int}} <: NLPModels.AbstractNLPModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    H_diag::VT
    A::MT
    g0::VT
    hrows::VI
    hcols::VI
    jrows::VI
    jcols::VI
end

function NLPModels.obj(qp::TwoStageQP{T}, x::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds for i in eachindex(x)
        s += 0.5 * qp.H_diag[i] * x[i] * x[i] + qp.g0[i] * x[i]
    end
    return s
end

function NLPModels.grad!(qp::TwoStageQP, x::AbstractVector, g::AbstractVector)
    @inbounds @simd for i in eachindex(x)
        g[i] = qp.H_diag[i] * x[i] + qp.g0[i]
    end
    return g
end

NLPModels.cons!(qp::TwoStageQP, x::AbstractVector, c::AbstractVector) = mul!(c, qp.A, x)

function NLPModels.jac_structure!(qp::TwoStageQP, I::AbstractVector{T}, J::AbstractVector{T}) where {T}
    copyto!(I, qp.jrows)
    copyto!(J, qp.jcols)
end

function NLPModels.jac_coord!(qp::TwoStageQP, x::AbstractVector, J::AbstractVector)
    @inbounds for k in eachindex(qp.jrows)
        J[k] = qp.A[qp.jrows[k], qp.jcols[k]]
    end
    return J
end

function NLPModels.hess_structure!(qp::TwoStageQP, I::AbstractVector{T}, J::AbstractVector{T}) where {T}
    copyto!(I, qp.hrows)
    copyto!(J, qp.hcols)
end

function NLPModels.hess_coord!(qp::TwoStageQP{T}, x::AbstractVector, l::AbstractVector, hess::AbstractVector; obj_weight = one(T)) where {T}
    @inbounds for k in eachindex(qp.hrows)
        hess[k] = obj_weight * qp.H_diag[qp.hrows[k]]
    end
    return hess
end

function build_twostage_qp(;
        ns::Int, nv::Int, nd::Int, nc::Int,
        hess_v::Matrix{T}, hess_d::Vector{T},
        g_v::Matrix{T}, g_d::Vector{T},
        A_v::Array{T, 3}, A_d::Array{T, 3},
        lcon::Matrix{T}, ucon::Matrix{T},
        lvar_v::Matrix{T}, uvar_v::Matrix{T},
        lvar_d::Vector{T}, uvar_d::Vector{T},
    ) where {T}
    n = ns * nv + nd
    m = ns * nc
    off = ns * nv

    x0 = zeros(T, n)
    lvar = Vector{T}(undef, n)
    uvar = Vector{T}(undef, n)
    g0 = Vector{T}(undef, n)
    H_diag = Vector{T}(undef, n)

    @inbounds for k in 1:ns, j in 1:nv
        i = (k - 1) * nv + j
        lvar[i] = lvar_v[j, k]
        uvar[i] = uvar_v[j, k]
        g0[i] = g_v[j, k]
        H_diag[i] = hess_v[j, k]
    end
    @inbounds for j in 1:nd
        i = off + j
        lvar[i] = lvar_d[j]
        uvar[i] = uvar_d[j]
        g0[i] = g_d[j]
        H_diag[i] = hess_d[j]
    end

    glcon = Vector{T}(undef, m)
    gucon = Vector{T}(undef, m)
    @inbounds for k in 1:ns, i in 1:nc
        idx = (k - 1) * nc + i
        glcon[idx] = lcon[i, k]
        gucon[idx] = ucon[i, k]
    end

    A = zeros(T, m, n)
    @inbounds for k in 1:ns, i in 1:nc
        row = (k - 1) * nc + i
        for j in 1:nv
            A[row, (k - 1) * nv + j] = A_v[i, j, k]
        end
        for j in 1:nd
            A[row, off + j] = A_d[i, j, k]
        end
    end

    nnzj = ns * nc * (nv + nd)
    jrows = Vector{Int}(undef, nnzj)
    jcols = Vector{Int}(undef, nnzj)
    p = 1
    @inbounds for k in 1:ns, i in 1:nc
        row = (k - 1) * nc + i
        for j in 1:nv
            jrows[p] = row; jcols[p] = (k - 1) * nv + j; p += 1
        end
        for j in 1:nd
            jrows[p] = row; jcols[p] = off + j; p += 1
        end
    end

    nnzh = n
    hrows = collect(1:n)
    hcols = collect(1:n)

    meta = NLPModels.NLPModelMeta(
        n;
        ncon = m,
        nnzj = nnzj,
        nnzh = nnzh,
        x0 = x0,
        y0 = zeros(T, m),
        lvar = lvar,
        uvar = uvar,
        lcon = glcon,
        ucon = gucon,
        minimize = true,
    )

    return TwoStageQP(
        meta, NLPModels.Counters(),
        H_diag, A, g0, hrows, hcols, jrows, jcols,
    )
end

# Convenience: kkt_options dict for SchurComplementKKTSystem.
schur_opts(; ns, nv, nd, nc) = Dict{Symbol, Any}(
    :schur_ns => ns, :schur_nv => nv, :schur_nd => nd, :schur_nc => nc,
)

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
