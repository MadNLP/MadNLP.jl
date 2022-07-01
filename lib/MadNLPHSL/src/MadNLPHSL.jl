module MadNLPHSL

import Libdl: dlopen, RTLD_DEEPBIND
import MadNLP: @kwdef, Logger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, set_options!, SparseMatrixCSC, SubVector,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, findIJ, nnz,
    get_tril_to_full, transfer!, input_type, _madnlp_unsafe_wrap

include(joinpath("..","deps","deps.jl"))

if @isdefined(libhsl)
    include("common.jl")
    include("mc68.jl")
    include("ma27.jl")
    include("ma57.jl")
    include("ma77.jl")
    include("ma86.jl")
    include("ma97.jl")
    export Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver
end

function __init__()
    check_deps()
    try
        @isdefined(libhsl) && dlopen(libhsl,RTLD_DEEPBIND)
    catch e
        println("HSL shared library cannot be loaded")
    end
end


end # module
