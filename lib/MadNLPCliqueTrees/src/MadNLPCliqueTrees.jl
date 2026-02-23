module MadNLPCliqueTrees

using CliqueTrees
using CliqueTrees: EliminationAlgorithm, SupernodeType
using CliqueTrees.Multifrontal
using CliqueTrees.Multifrontal: FChordalLDLt, FChordalCholesky
using CliqueTrees: DEFAULT_ELIMINATION_ALGORITHM, DEFAULT_SUPERNODE_TYPE
using LinearAlgebra
using LinearAlgebra: PivotingStrategy
using SparseArrays

import MadNLP:
    MadNLP,
    @kwdef,
    MadNLPLogger,
    AbstractOptions,
    AbstractLinearSolver,
    LinearFactorization,
    LDL,
    CHOLESKY,
    set_options!,
    introduce,
    factorize!,
    solve_linear_system!,
    improve!,
    is_inertia,
    inertia,
    input_type,
    is_supported,
    default_options

@kwdef mutable struct CliqueTreesOptions <: AbstractOptions
    cliquetrees_algorithm::LinearFactorization = LDL
    cliquetrees_ordering::EliminationAlgorithm = DEFAULT_ELIMINATION_ALGORITHM # AMF
    cliquetrees_supernode::SupernodeType = DEFAULT_SUPERNODE_TYPE # Maximal
    cliquetrees_strategy::PivotingStrategy = NoPivot()
end

mutable struct CliqueTreesSolver{T, F <: Factorization{T}} <: AbstractLinearSolver{T}
    tril::SparseMatrixCSC{T, Int32}
    F::F
    opt::CliqueTreesOptions
    logger::MadNLPLogger
end

function _build_factorization(tril::SparseMatrixCSC{T, Int32}, opt::CliqueTreesOptions, ::Val{LDL}) where T
    S = Symmetric(tril, :L)
    F = ChordalLDLt{:L}(S; alg = opt.cliquetrees_ordering, snd = opt.cliquetrees_supernode)::FChordalLDLt{:L, T, Int32}
    return F
end

function _build_factorization(tril::SparseMatrixCSC{T, Int32}, opt::CliqueTreesOptions, ::Val{CHOLESKY}) where T
    S = Symmetric(tril, :L)
    F = ChordalCholesky{:L}(S; alg = opt.cliquetrees_ordering, snd = opt.cliquetrees_supernode)::FChordalCholesky{:L, T, Int32}
    return F
end

function CliqueTreesSolver(
    tril::SparseMatrixCSC{T, Int32};
    opt = CliqueTreesOptions(),
    logger = MadNLPLogger(),
) where T
    F = _build_factorization(tril, opt, Val(opt.cliquetrees_algorithm))
    return CliqueTreesSolver{T, typeof(F)}(tril, F, opt, logger)
end

function factorize!(M::CliqueTreesSolver{T, <:ChordalLDLt}) where T
    ldlt!(copy!(M.F, M.tril), M.opt.cliquetrees_strategy; check=false)
    return M
end

function solve_linear_system!(M::CliqueTreesSolver{T, <:ChordalLDLt}, rhs::Vector{T}) where T
    if issuccess(M.F)
        ldiv!(M.F, rhs)
    end

    return rhs
end

function inertia(M::CliqueTreesSolver{T, <:ChordalLDLt}) where T
    d = M.F.d
    pos = 0; zer = 0; neg = 0
    @inbounds for di in d
        if di > 0
            pos += 1
        elseif di == 0
            zer += 1
        else
            neg += 1
        end
    end
    return pos, zer, neg
end

introduce(::CliqueTreesSolver{T, <:ChordalLDLt}) where T =
    "CliqueTrees/LDLᵀ v$(pkgversion(CliqueTrees))"

function factorize!(M::CliqueTreesSolver{T, <:ChordalCholesky}) where T
    cholesky!(copy!(M.F, M.tril), M,opt.cliquetrees_strategy; check=false)
    return M
end

function solve_linear_system!(M::CliqueTreesSolver{T, <:ChordalCholesky}, rhs::Vector{T}) where T
    if issuccess(M.F)
        ldiv!(M.F, rhs)
    end

    return rhs
end

function inertia(M::CliqueTreesSolver{T, <:ChordalCholesky}) where T
    n = size(M.tril, 1)

    if issuccess(M.F)
        return (n, 0, 0)
    else
        return (0, n, 0)
    end
end

introduce(::CliqueTreesSolver{T, <:ChordalCholesky}) where T =
    "CliqueTrees/Cholesky v$(pkgversion(CliqueTrees))"

is_inertia(::CliqueTreesSolver) = true
improve!(::CliqueTreesSolver) = false
input_type(::Type{<:CliqueTreesSolver}) = :csc
default_options(::Type{<:CliqueTreesSolver}) = CliqueTreesOptions()
is_supported(::Type{<:CliqueTreesSolver}, ::Type{T}) where T <: AbstractFloat = true

export CliqueTreesSolver, CliqueTreesOptions

for name in names(MadNLP, all = true)
    if Base.isexported(MadNLP, name)
        @eval using MadNLP: $(name)
        @eval export $(name)
    end
end

end # module MadNLPCliqueTrees
