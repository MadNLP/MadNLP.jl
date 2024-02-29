using LDLFactorizations

@kwdef mutable struct LDLFactorizationsOptions <: AbstractOptions
end

mutable struct LDLSolver{T} <: AbstractLinearSolver{T}
    inner::LDLFACTORIZATIONS.LDLFactorizationsLU{Float64, Int32}
    tril::SparseMatrixCSC{T,Int32}
    opt::LDLFactorizationsOptions
    logger::MadNLPLogger
end

function LDLSolver(
    csc::SparseMatrixCSC{T};
    opt=LDLFactorizationsOptions(), logger=MadNLPLogger(),
) where T
    
    return LDLSolver(
        ldl(
            Symmetric(tril, :L)
        ),
        tril, opt, logger
    )
end

function factorize!(M::LDLSolver)
    ldl_factorize!(M.tril, M.inner)
    return M
end

function solve!(M::LDLSolver{T},rhs::Vector{T}) where T
    ldiv!(M.inner, rhs)
    # If the factorization failed, we return the same
    # rhs to enter into a primal-dual regularization phase.
    return rhs
end

is_inertia(::LDLSolver) = true
inertia(M::LDLSolver) = M.issucsses ? (size(M.tril,1),0,0) : (size(M.tril,1)-2,1,1)
input_type(::Type{LDLSolver}) = :csc
default_options(::Type{LDLSolver}) = LDLFactorizationsOptions()

function improve!(M::LDLSolver)
    return false
end
introduce(::LDLSolver) = "LDLFactorizations"
is_supported(::Type{LDLSolver},::Type{Float32}) = true
is_supported(::Type{LDLSolver},::Type{Float64}) = true
