struct HS15Model{T, VT <: AbstractVector{T}} <: NLPModels.AbstractNLPModel{T,VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
end

function HS15Model(;T = Float64, x0=zeros(T,2))
    return HS15Model(
        NLPModels.NLPModelMeta(
            2,     #nvar
            ncon = 2,
            nnzj = 4,
            nnzh = 3,
            x0 = x0,
            y0 = copyto!(similar(x0,2), (0.0, 0.0)),
            lvar = copyto!(similar(x0,2), (-Inf, -Inf)),
            uvar = copyto!(similar(x0,2), (0.5, Inf)),
            lcon = copyto!(similar(x0,2), (1.0, 0.0)),
            ucon = copyto!(similar(x0,2), (Inf, Inf)),
            minimize = true
        ),
        NLPModels.Counters()
    )
end

function NLPModels.obj(nlp::HS15Model{T}, x::AbstractVector{T}) where T
    return (T(100.0) * (x[2] - x[1]^2)^2 + (one(T) - x[1])^2)::T
end

function NLPModels.grad!(nlp::HS15Model{T}, x::AbstractVector, g::AbstractVector) where T
    z = x[2] - x[1]^2
    g[1] = -T(400.0) * z * x[1] - 2.0 * (T(1.0) - x[1])
    g[2] = T(200.0) * z
    return g
end

function NLPModels.cons!(nlp::HS15Model, x::AbstractVector, c::AbstractVector) 
    c[1] = x[1] * x[2]
    c[2] = x[1] + x[2]^2
    return c
end

function NLPModels.jac_structure!(nlp::HS15Model, I::AbstractVector, J::AbstractVector)
    copyto!(I, [1, 1, 2, 2])
    copyto!(J, [1, 2, 1, 2])
end

function NLPModels.jac_coord!(nlp::HS15Model{T}, x::AbstractVector, J::AbstractVector) where T
    J[1] = x[2]    # (1, 1)
    J[2] = x[1]    # (1, 2)
    J[3] = T(1.0)  # (2, 1)
    J[4] = 2*x[2]  # (2, 2)
    return J
end

function NLPModels.jprod!(nlp::HS15Model{T}, x::AbstractVector, v::AbstractVector, jv::AbstractVector) where T
    jv[1] = x[2] * v[1] + x[1] * v[2]
    jv[2] = v[1] + 2 * x[2] * v[2]
    return jv
end

function NLPModels.jtprod!(nlp::HS15Model{T}, x::AbstractVector, v::AbstractVector, jv::AbstractVector) where T
    jv[1] = x[2] * v[1] + v[2]
    jv[2] = x[1] * v[1] + 2 * x[2] * v[2]
    return jv
end

function NLPModels.hess_structure!(nlp::HS15Model, I::AbstractVector, J::AbstractVector)
    copyto!(I, [1, 2, 2])
    copyto!(J, [1, 1, 2])
end

function NLPModels.hess_coord!(nlp::HS15Model{T}, x::AbstractVector{T}, y::AbstractVector{T}, H::AbstractVector{T}; obj_weight::T=one(T)) where T
    # Objective
    H[1] = obj_weight * (T(-400.0) * x[2] + T(1200.0) * x[1]^2 + T(2.0))
    H[2] = obj_weight * (T(-400.0) * x[1])
    H[3] = obj_weight * T(200.0)
    # First constraint
    H[2] += y[1] * T(1.0)
    # Second constraint
    H[3] += y[2] * T(2.0)
    return H
end


@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.

    __init__()
    
    @compile_workload begin
        nlp = HS15Model()
        madnlp(nlp; print_level=MadNLP.ERROR)
        
        precompile(Tuple{typeof(MadNLP.madnlp), NLPModels.AbstractNLPModel{T, S} where S where T})
        precompile(Tuple{Type{NamedTuple{(:color,), T} where T<:Tuple}, Tuple{Int64}})
        precompile(Tuple{typeof(Base.argtail), Float64, Float64, Vararg{Float64}})
        precompile(Tuple{typeof(MadNLP.get_status_output), MadNLP.Status, MadNLP.MadNLPOptions{Float64}})
        precompile(Tuple{typeof(MadNLP.improve!), MadNLP.MumpsSolver{Float64}})
        precompile(Tuple{typeof(MadNLP.introduce), MadNLP.MumpsSolver{Float64}})
        precompile(Tuple{typeof(Base.indexed_iterate), Pair{Symbol, Float64}, Int64})
        precompile(Tuple{typeof(Base.indexed_iterate), Pair{Symbol, Float64}, Int64, Int64})
        precompile(Tuple{typeof(Base.round), Float64})                                                                               
        precompile(Tuple{typeof(Core.kwcall), NamedTuple{names, T} where T<:Tuple where names, Type{MadNLP.MadNLPOptions{Float64}}, NLPModels.AbstractNLPModel{T, S} where S where T})
    end
end

