# AOT Compilation Example for MadNLP
#
# This file demonstrates ahead-of-time compilation of MadNLP using Julia's
# native compilation capabilities. It defines a self-contained entry point
# that solves a small nonlinear program (Hock-Schittkowski problem #15).
#
# Usage with juliac (Julia 1.12+):
#   julia --project=aot juliac.jl --output-exe madnlp_solve aot_example.jl
#
# Usage with PackageCompiler.jl:
#   See aot_sysimage.jl for creating a system image.
#
# The entry point avoids all AOT-incompatible patterns:
#   - No eval() at runtime
#   - No invokelatest()
#   - No setglobal!() at runtime
#   - No string-to-type parsing

module MadNLPAOTExample

using MadNLP
using NLPModels

# ─── Define the HS15 problem ───────────────────────────────────────────
struct HS15Model{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
end

function HS15Model(; T=Float64, x0=zeros(T, 2), y0=zeros(T, 2))
    return HS15Model(
        NLPModels.NLPModelMeta(
            2;
            ncon=2, nnzj=4, nnzh=3,
            x0=x0, y0=y0,
            lvar=T[-Inf, -Inf], uvar=T[0.5, Inf],
            lcon=T[1.0, 0.0], ucon=T[Inf, Inf],
            minimize=true,
        ),
        NLPModels.Counters(),
    )
end

function NLPModels.obj(::HS15Model, x::AbstractVector)
    return 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
end

function NLPModels.grad!(::HS15Model, x::AbstractVector, g::AbstractVector)
    z = x[2] - x[1]^2
    g[1] = -400.0 * z * x[1] - 2.0 * (1.0 - x[1])
    g[2] = 200.0 * z
    return g
end

function NLPModels.cons!(::HS15Model, x::AbstractVector, c::AbstractVector)
    c[1] = x[1] * x[2]
    c[2] = x[1] + x[2]^2
    return c
end

function NLPModels.jac_structure!(::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 1, 2, 2])
    copyto!(J, [1, 2, 1, 2])
    return (I, J)
end

function NLPModels.jac_coord!(::HS15Model, x::AbstractVector, J::AbstractVector)
    J[1] = x[2]; J[2] = x[1]; J[3] = 1.0; J[4] = 2 * x[2]
    return J
end

function NLPModels.hess_structure!(::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 2, 2])
    copyto!(J, [1, 1, 2])
    return (I, J)
end

function NLPModels.hess_coord!(::HS15Model, x, y, H::AbstractVector; obj_weight=1.0)
    H[1] = obj_weight * (-400.0 * x[2] + 1200.0 * x[1]^2 + 2.0)
    H[2] = obj_weight * (-400.0 * x[1])
    H[3] = obj_weight * 200.0
    H[2] += y[1] * 1.0
    H[3] += y[2] * 2.0
    return H
end

# ─── AOT-safe entry point ─────────────────────────────────────────────

function solve_hs15()
    nlp = HS15Model()
    # All options passed as concrete types, no string parsing
    result = madnlp(nlp; print_level=MadNLP.INFO)
    return result
end

# ─── Main for standalone execution ────────────────────────────────────

function julia_main()::Cint
    try
        result = solve_hs15()
        println("Status: ", result.status)
        println("Objective: ", result.objective)
        println("Solution: ", result.solution)
        return 0
    catch e
        println(stderr, "Error: ", e)
        return 1
    end
end

end # module

# When run as a script
if abspath(PROGRAM_FILE) == @__FILE__
    exit(MadNLPAOTExample.julia_main())
end
