module MadNLPPardiso

include(joinpath("..","deps","deps.jl"))

const INPUT_MATRIX_TYPE = :csc

import Libdl: dlopen, RTLD_DEEPBIND
import MadNLP:
    MadNLP, @kwdef, Logger, @debug, @warn, @error,
    SubVector, StrideOneVector, SparseMatrixCSC, 
    SymbolicException,FactorizationException,SolveException,InertiaException,
    AbstractOptions, AbstractLinearSolver, set_options!,
    introduce, factorize!, solve!, improve!, is_inertia, inertia
import MKL_jll: libmkl_rt

@isdefined(libpardiso) && include("pardiso.jl")
include("pardisomkl.jl")

function __init__()
    check_deps()
    @isdefined(libpardiso) && dlopen(libpardiso,RTLD_DEEPBIND)
end

export MadNLPPardisoMKL

end # module
