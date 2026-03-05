using Test
using LinearAlgebra
using CUDA
using CUDSS
using MadNLP
using MadNLPGPU
import MadNLP.NLPModels

# Hand-rolled NLPModel for two-stage stochastic QPs used to exercise
# GPUSchurComplementKKTSystem without pulling in ExaModels.
#
# Variable layout:   x = [v_{1,1},...,v_{1,nv}, v_{2,1},...,v_{ns,nv}, d_1,...,d_nd]
# Constraint layout: c = [c_{1,1},...,c_{1,nc}, c_{2,1},...,c_{ns,nc}]
# Hessian:  diagonal.
# Jacobian: per-scenario block-coupled.

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
    MadNLPGPU.@allowscalar for i in eachindex(x)
        s += 0.5 * qp.H_diag[i] * x[i] * x[i] + qp.g0[i] * x[i]
    end
    return s
end

function NLPModels.grad!(qp::TwoStageQP, x::AbstractVector, g::AbstractVector)
    g .= qp.H_diag .* x .+ qp.g0
    return g
end

NLPModels.cons!(qp::TwoStageQP, x::AbstractVector, c::AbstractVector) = mul!(c, qp.A, x)

function NLPModels.jac_structure!(qp::TwoStageQP, I::AbstractVector{T}, J::AbstractVector{T}) where {T}
    copyto!(I, qp.jrows)
    copyto!(J, qp.jcols)
end

function NLPModels.jac_coord!(qp::TwoStageQP, x::AbstractVector, J::AbstractVector)
    MadNLPGPU.@allowscalar for k in eachindex(qp.jrows)
        J[k] = qp.A[qp.jrows[k], qp.jcols[k]]
    end
    return J
end

function NLPModels.hess_structure!(qp::TwoStageQP, I::AbstractVector{T}, J::AbstractVector{T}) where {T}
    copyto!(I, qp.hrows)
    copyto!(J, qp.hcols)
end

function NLPModels.hess_coord!(qp::TwoStageQP{T}, x::AbstractVector, l::AbstractVector, hess::AbstractVector; obj_weight = one(T)) where {T}
    MadNLPGPU.@allowscalar for k in eachindex(qp.hrows)
        hess[k] = obj_weight * qp.H_diag[qp.hrows[k]]
    end
    return hess
end

# Backend-aware builder. `x0_template` selects the storage device:
# CPU `Vector{T}` or `CuVector{T}` etc.
function build_twostage_qp(
        x0_template::AbstractVector{T};
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

    # Build everything on CPU then move to device at the end.
    x0_h = zeros(T, n)
    lvar_h = Vector{T}(undef, n)
    uvar_h = Vector{T}(undef, n)
    g0_h = Vector{T}(undef, n)
    H_diag_h = Vector{T}(undef, n)

    for k in 1:ns, j in 1:nv
        i = (k - 1) * nv + j
        lvar_h[i] = lvar_v[j, k]
        uvar_h[i] = uvar_v[j, k]
        g0_h[i] = g_v[j, k]
        H_diag_h[i] = hess_v[j, k]
    end
    for j in 1:nd
        i = off + j
        lvar_h[i] = lvar_d[j]
        uvar_h[i] = uvar_d[j]
        g0_h[i] = g_d[j]
        H_diag_h[i] = hess_d[j]
    end

    glcon_h = Vector{T}(undef, m)
    gucon_h = Vector{T}(undef, m)
    for k in 1:ns, i in 1:nc
        idx = (k - 1) * nc + i
        glcon_h[idx] = lcon[i, k]
        gucon_h[idx] = ucon[i, k]
    end

    A_h = zeros(T, m, n)
    for k in 1:ns, i in 1:nc
        row = (k - 1) * nc + i
        for j in 1:nv
            A_h[row, (k - 1) * nv + j] = A_v[i, j, k]
        end
        for j in 1:nd
            A_h[row, off + j] = A_d[i, j, k]
        end
    end

    nnzj = ns * nc * (nv + nd)
    jrows_h = Vector{Int}(undef, nnzj)
    jcols_h = Vector{Int}(undef, nnzj)
    p = 1
    for k in 1:ns, i in 1:nc
        row = (k - 1) * nc + i
        for j in 1:nv
            jrows_h[p] = row; jcols_h[p] = (k - 1) * nv + j; p += 1
        end
        for j in 1:nd
            jrows_h[p] = row; jcols_h[p] = off + j; p += 1
        end
    end

    hrows_h = collect(1:n)
    hcols_h = collect(1:n)

    # Move to the storage device implied by x0_template.
    x0 = similar(x0_template, n); copyto!(x0, x0_h)
    lvar = similar(x0_template, n); copyto!(lvar, lvar_h)
    uvar = similar(x0_template, n); copyto!(uvar, uvar_h)
    g0 = similar(x0_template, n); copyto!(g0, g0_h)
    H_diag = similar(x0_template, n); copyto!(H_diag, H_diag_h)
    A = similar(x0_template, m, n); copyto!(A, A_h)
    glcon = similar(x0_template, m); copyto!(glcon, glcon_h)
    gucon = similar(x0_template, m); copyto!(gucon, gucon_h)
    y0 = similar(x0_template, m); fill!(y0, zero(T))

    nnzh = n
    hrows = jrows_h isa Vector ? hrows_h : copyto!(similar(x0_template, Int, nnzh), hrows_h)
    hcols = jcols_h isa Vector ? hcols_h : copyto!(similar(x0_template, Int, nnzh), hcols_h)
    jrows = jrows_h isa Vector ? jrows_h : copyto!(similar(x0_template, Int, nnzj), jrows_h)
    jcols = jcols_h isa Vector ? jcols_h : copyto!(similar(x0_template, Int, nnzj), jcols_h)

    meta = NLPModels.NLPModelMeta(
        n;
        ncon = m,
        nnzj = nnzj,
        nnzh = nnzh,
        x0 = x0,
        y0 = y0,
        lvar = lvar,
        uvar = uvar,
        lcon = glcon,
        ucon = gucon,
        minimize = true,
    )

    return TwoStageQP(
        meta, NLPModels.Counters(),
        H_diag, A, g0, hrows_h, hcols_h, jrows_h, jcols_h,
    )
end

schur_opts(; ns, nv, nd, nc) = Dict{Symbol, Any}(
    :schur_ns => ns, :schur_nv => nv, :schur_nd => nd, :schur_nc => nc,
)

@testset "GPUSchurComplementKKTSystem" begin

    @testset "Basic convergence — quadratic with coupling" begin
        ns, nv, nd, nc = 3, 1, 1, 1
        θ = [4.0, 6.0, 8.0]

        nlp = build_twostage_qp(
            CUDA.zeros(Float64, ns * nv + nd);
            ns, nv, nd, nc,
            hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
            g_v = reshape(-2 .* θ, nv, ns), g_d = [-2.0],
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = zeros(nc, ns), ucon = zeros(nc, ns),
            lvar_v = fill(-100.0, nv, ns), uvar_v = fill(100.0, nv, ns),
            lvar_d = fill(-100.0, nd), uvar_d = fill(100.0, nd),
        )

        result = madnlp(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCUDASolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Match CPU reference" begin
        ns, nv, nd, nc = 2, 2, 1, 1
        θ = [1.0 3.0; 2.0 4.0]

        common = (
            ns = ns, nv = nv, nd = nd, nc = nc,
            hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
            g_v = -2 .* θ, g_d = [0.0],
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = zeros(nc, ns), ucon = zeros(nc, ns),
            lvar_v = fill(-50.0, nv, ns), uvar_v = fill(50.0, nv, ns),
            lvar_d = fill(-50.0, nd), uvar_d = fill(50.0, nd),
        )

        nlp_cpu = build_twostage_qp(zeros(Float64, ns * nv + nd); common...)
        nlp_gpu = build_twostage_qp(CUDA.zeros(Float64, ns * nv + nd); common...)

        ref = madnlp(
            nlp_cpu;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCPUSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        gpu_result = madnlp(
            nlp_gpu;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCUDASolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )

        @test ref.status == MadNLP.SOLVE_SUCCEEDED
        @test gpu_result.status == MadNLP.SOLVE_SUCCEEDED
        @test isapprox(gpu_result.objective, ref.objective; atol = 1.0e-6)
        @test isapprox(Array(gpu_result.solution), Array(ref.solution); atol = 1.0e-4)
    end

    @testset "Multiple recourse vars and design vars" begin
        ns, nv, nd, nc = 2, 2, 2, 1
        θ = [1.0, 2.0]
        H_v = zeros(nv, ns)
        for k in 1:ns, j in 1:nv
            H_v[j, k] = 2θ[k]
        end

        nlp = build_twostage_qp(
            CUDA.zeros(Float64, ns * nv + nd);
            ns, nv, nd, nc,
            hess_v = H_v, hess_d = fill(2.0, nd),
            g_v = zeros(nv, ns), g_d = zeros(nd),
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = ones(nc, ns), ucon = ones(nc, ns),
            lvar_v = fill(-50.0, nv, ns), uvar_v = fill(50.0, nv, ns),
            lvar_d = fill(-50.0, nd), uvar_d = fill(50.0, nd),
        )

        result = madnlp(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCUDASolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Known solution with inactive constraints" begin
        ns, nv, nd, nc = 2, 1, 1, 1
        θ = [3.0, 7.0]

        nlp = build_twostage_qp(
            CUDA.zeros(Float64, ns * nv + nd);
            ns, nv, nd, nc,
            hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
            g_v = reshape(-2 .* θ, nv, ns), g_d = [-10.0],
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = fill(-100.0, nc, ns), ucon = fill(100.0, nc, ns),
            lvar_v = fill(-100.0, nv, ns), uvar_v = fill(100.0, nv, ns),
            lvar_d = fill(-100.0, nd), uvar_d = fill(100.0, nd),
        )

        result = madnlp(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCUDASolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        sol = Array(result.solution)
        @test isapprox(sol[1], 3.0; atol = 1.0e-3)
        @test isapprox(sol[2], 7.0; atol = 1.0e-3)
        @test isapprox(sol[3], 5.0; atol = 1.0e-3)
    end
end
