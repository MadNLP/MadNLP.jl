@kwdef mutable struct UmfpackOptions <: AbstractOptions
    umfpack_pivtol::Float64 = 1e-4
    umfpack_pivtolmax::Float64 = 1e-1
    umfpack_sym_pivtol::Float64 = 1e-3
    umfpack_block_size::Float64 = 16
    umfpack_strategy::Float64 = 2.
end

mutable struct UmfpackSolver{T} <: AbstractLinearSolver{T}
    inner::UMFPACK.UmfpackLU{T, Int32}
    tril::SparseMatrixCSC{T}
    full::SparseMatrixCSC{T}
    tril_to_full_view::SubVector{T}

    p::Vector{T}

    opt::UmfpackOptions
    logger::MadNLPLogger
end

function UmfpackSolver(
    csc::SparseMatrixCSC{T};
    opt=UmfpackOptions(), logger=MadNLPLogger(),
) where T
    p = Vector{T}(undef,csc.n)
    full, tril_to_full_view = get_tril_to_full(csc)
    controls = UMFPACK.get_umfpack_control(T, Int)
    # Override default controls with custom setting
    controls[4] = opt.umfpack_pivtol
    controls[5] = opt.umfpack_block_size
    controls[6] = opt.umfpack_strategy
    controls[12] = opt.umfpack_sym_pivtol
    inner = UMFPACK.UmfpackLU(csc; control=controls)
    return UmfpackSolver(inner, csc, full, tril_to_full_view, p, opt, logger)
end

function factorize!(M::UmfpackSolver)
    M.full.nzval .= M.tril_to_full_view
    # We check the factorization succeeded later in the backsolve
    UMFPACK.lu!(M.inner, M.full; check=false)
    return M
end

function solve!(M::UmfpackSolver{T},rhs::Vector{T}) where T
    if UMFPACK.issuccess(M.inner)
        UMFPACK.ldiv!(M.p, M.inner, rhs)
        rhs .= M.p
    end
    # If the factorization failed, we return the same
    # rhs to enter into a primal-dual regularization phase.
    return rhs
end

is_inertia(::UmfpackSolver) = false
inertia(M::UmfpackSolver) = throw(InertiaException())
input_type(::Type{UmfpackSolver}) = :csc
default_options(::Type{UmfpackSolver}) = UmfpackOptions()

function improve!(M::UmfpackSolver)
    if M.inner.control[4] == M.opt.umfpack_pivtolmax
        @debug(M.logger, "improve quality failed.")
        return false
    end
    M.inner.control[4] = min(M.opt.umfpack_pivtolmax, M.inner.control[4]^.75)
    @debug(M.logger, "improved quality: pivtol = $(M.inner.control[4])")
    return true
end
introduce(::UmfpackSolver) = "umfpack"
is_supported(::Type{UmfpackSolver},::Type{Float64}) = true
