# TODO: check options in CHOLMOD
@kwdef mutable struct CHOLMODOptions <: AbstractOptions
end

mutable struct CHOLMODSolver{T} <: AbstractLinearSolver{T}
    inner::CHOLMOD.Factor{Float64}
    tril::SparseMatrixCSC{T,Int32}
    full::SparseMatrixCSC{Float64,Int32}
    tril_to_full_view::SubVector{T}

    p::Vector{Float64}
    d::Vector{Float64}

    opt::CHOLMODOptions
    logger::MadNLPLogger
end

function CHOLMODSolver(
    csc::SparseMatrixCSC{T};
    opt=CHOLMODOptions(), logger=MadNLPLogger(),
) where T
    p = Vector{Float64}(undef,csc.n)
    d = Vector{Float64}(undef,csc.n)
    full, tril_to_full_view = get_tril_to_full(Float64,csc)
    full.nzval .= 1.0

    A = CHOLMOD.Sparse(full)
    # TODO: use AMD permutation here
    inner = CHOLMOD.symbolic(A)

    return CHOLMODSolver(inner, csc, full, tril_to_full_view, p, d, opt, logger)
end

function factorize!(M::CHOLMODSolver)
    M.full.nzval .= M.tril_to_full_view
    # We check the factorization succeeded later in the backsolve
    CHOLMOD.cholesky!(M.inner, M.full; check=false)
    return M
end

function solve!(M::CHOLMODSolver{T}, rhs::Vector{T}) where T
    if issuccess(M.inner)
        B = CHOLMOD.Dense(rhs)
        X = CHOLMOD.solve(CHOLMOD.CHOLMOD_A, M.inner, B)
        copyto!(rhs, X)
    end
    # If the factorization failed, we return the same
    # rhs to enter into a primal-dual regularization phase.
    return rhs
end

is_inertia(::CHOLMODSolver) = true
function inertia(M::CHOLMODSolver)
    n = size(M.full, 1)
    if issuccess(M.inner)
        return (n, 0, 0)
    else
        return (0, n, 0)
    end
end
input_type(::Type{CHOLMODSolver}) = :csc
default_options(::Type{CHOLMODSolver}) = CHOLMODOptions()

improve!(M::CHOLMODSolver) = false
introduce(::CHOLMODSolver) = "cholmod"
is_supported(::Type{CHOLMODSolver},::Type{Float32}) = true
is_supported(::Type{CHOLMODSolver},::Type{Float64}) = true
