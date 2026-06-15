module MadCorePardiso

import Libdl: dlopen, RTLD_DEEPBIND

# Locate the proprietary Panua PARDISO shared library at load time via the
# JULIA_PARDISO env var (same lookup as upstream MadNLPPardiso's deps/build.jl,
# but done at load instead of a separate build step). Empty when absent:
# PardisoSolver is still defined but unusable; PardisoMKLSolver (via MKL_jll)
# always works.
const _PARDISO_NAMES = Sys.iswindows() ? ["libpardiso.dll", "libpardiso600-WIN-X86-64.dll"] :
    Sys.isapple() ? ["libpardiso.dylib", "libpardiso600-MACOS-X86-64.dylib"] :
    ["libpardiso.so", "libpardiso600-GNU800-X86-64.so"]
function _find_pardiso()
    for prefix in split(get(ENV, "JULIA_PARDISO", ""), ':'; keepempty = false)
        for name in _PARDISO_NAMES
            path = joinpath(prefix, name)
            isfile(path) && return path
        end
    end
    return ""
end
const libpardiso = _find_pardiso()

import MadCore:
    @kwdef,
    MadNLPLogger,
    @debug,
    @warn,
    @error,
    SubVector,
    SparseMatrixCSC,
    SymbolicException,
    FactorizationException,
    SolveException,
    InertiaException,
    AbstractOptions,
    AbstractLinearSolver,
    LinearFactorization,
    BUNCHKAUFMAN,
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
    blas_num_threads,
    is_supported,
    default_options
import MKL_jll: libmkl_rt

# The proprietary wrapper + solver compile even with an empty `libpardiso`
# (the ccalls resolve at call time), matching upstream behavior.
include("libpardiso.jl")
include("pardiso.jl")
include("pardisomkl.jl")

function __init__()
    isempty(libpardiso) || dlopen(libpardiso, RTLD_DEEPBIND)
    return
end

export PardisoSolver, PardisoMKLSolver

end # module
