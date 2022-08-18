module MadNLPPardiso

include(joinpath("..","deps","deps.jl"))


import Libdl: dlopen, RTLD_DEEPBIND
import MadNLP:
    MadNLP, @kwdef, MadNLPLogger, @debug, @warn, @error,
    SubVector, SparseMatrixCSC,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    AbstractOptions, AbstractLinearSolver, set_options!,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, input_type,
    blas_num_threads, is_supported, default_options
import MKL_jll: libmkl_rt

@isdefined(libpardiso) && include("pardiso.jl")
include("pardisomkl.jl")

function __init__()
    check_deps()
    @isdefined(libpardiso) && dlopen(libpardiso,RTLD_DEEPBIND)
end

export PardisoSolver, PardisoMKLSolver

end # module
