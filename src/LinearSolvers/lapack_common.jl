# Shared methods for all LAPACK-based dense linear solvers.
# Each backend (CPU, CUDA, ROCm, oneAPI) subtypes AbstractLapackSolver
# and provides per-algorithm primitives (setup_*/factorize_*/solve_*).

# Per-algorithm primitives that each backend must implement.
function setup_cholesky! end
function setup_lu! end
function setup_qr! end
function setup_evd! end
function setup_bunchkaufman! end
function factorize_cholesky! end
function factorize_lu! end
function factorize_qr! end
function factorize_evd! end
function factorize_bunchkaufman! end
function solve_cholesky! end
function solve_lu! end
function solve_qr! end
function solve_evd! end
function solve_bunchkaufman! end

"""
    transfer_matrix!(M::AbstractLapackSolver)

Copy the matrix `M.A` into the factorization buffer `M.fact`.
GPU backends override this to use `gpu_transfer!`.
"""
transfer_matrix!(M::AbstractLapackSolver) = copyto!(M.fact, M.A)

"""
    supports_bunchkaufman_inertia(M::AbstractLapackSolver)

Return `true` if the solver supports inertia computation for Bunch-Kaufman.
Only the CPU backend overrides this to `true`.
"""
supports_bunchkaufman_inertia(::AbstractLapackSolver) = false

"""
    _get_info(M::AbstractLapackSolver)

Return the scalar info value from the factorization.
CPU uses `M.info[]` (Ref); GPU backends override to `sum(M.info)`.
"""
_get_info(M::AbstractLapackSolver) = M.info[]

function setup!(M::AbstractLapackSolver)
    if M.opt.lapack_algorithm == BUNCHKAUFMAN
        setup_bunchkaufman!(M)
    elseif M.opt.lapack_algorithm == LU
        setup_lu!(M)
    elseif M.opt.lapack_algorithm == QR
        setup_qr!(M)
    elseif M.opt.lapack_algorithm == CHOLESKY
        setup_cholesky!(M)
    elseif M.opt.lapack_algorithm == EVD
        setup_evd!(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

function factorize!(M::AbstractLapackSolver)
    transfer_matrix!(M)
    if M.opt.lapack_algorithm == BUNCHKAUFMAN
        factorize_bunchkaufman!(M)
    elseif M.opt.lapack_algorithm == LU
        tril_to_full!(M.fact)
        factorize_lu!(M)
    elseif M.opt.lapack_algorithm == QR
        tril_to_full!(M.fact)
        factorize_qr!(M)
    elseif M.opt.lapack_algorithm == CHOLESKY
        factorize_cholesky!(M)
    elseif M.opt.lapack_algorithm == EVD
        factorize_evd!(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

"""
    _solve_dispatch!(M::AbstractLapackSolver, x)

Dispatch `solve!` to the correct per-algorithm solve method.
"""
function _solve_dispatch!(M::AbstractLapackSolver, x)
    if M.opt.lapack_algorithm == BUNCHKAUFMAN
        solve_bunchkaufman!(M, x)
    elseif M.opt.lapack_algorithm == LU
        solve_lu!(M, x)
    elseif M.opt.lapack_algorithm == QR
        solve_qr!(M, x)
    elseif M.opt.lapack_algorithm == CHOLESKY
        solve_cholesky!(M, x)
    elseif M.opt.lapack_algorithm == EVD
        solve_evd!(M, x)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

function solve_linear_system!(M::AbstractLapackSolver, x::AbstractVector)
    isempty(M.sol) && resize!(M.sol, M.n)
    copyto!(M.sol, x)
    _solve_dispatch!(M, M.sol)
    copyto!(x, M.sol)
    return x
end

improve!(M::AbstractLapackSolver) = false
input_type(::Type{<:AbstractLapackSolver}) = :dense
default_options(::Type{<:AbstractLapackSolver}) = LapackOptions()

for T in (:Float32, :Float64)
    @eval is_supported(::Type{<:AbstractLapackSolver}, ::Type{$T}) = true
end

function is_inertia(M::AbstractLapackSolver)
    alg = M.opt.lapack_algorithm
    return (alg == CHOLESKY) ||
           (alg == EVD) ||
           (alg == BUNCHKAUFMAN && supports_bunchkaufman_inertia(M))
end

function inertia(M::AbstractLapackSolver)
    if M.opt.lapack_algorithm == CHOLESKY
        return inertia_cholesky(M)
    elseif M.opt.lapack_algorithm == EVD
        return inertia_evd(M)
    elseif M.opt.lapack_algorithm == BUNCHKAUFMAN && supports_bunchkaufman_inertia(M)
        return inertia_bunchkaufman(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

function inertia_cholesky(M::AbstractLapackSolver)
    return _get_info(M) == 0 ? (M.n, 0, 0) : (0, M.n, 0)
end

function inertia_evd(M::AbstractLapackSolver)
    numpos = count(λ -> λ > 0, M.Λ)
    numneg = count(λ -> λ < 0, M.Λ)
    numzero = M.n - numpos - numneg
    return (numpos, numzero, numneg)
end
