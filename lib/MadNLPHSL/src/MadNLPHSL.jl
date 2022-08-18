module MadNLPHSL

import Libdl: dlopen, RTLD_DEEPBIND
import MadNLP: @kwdef, Logger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, set_options!, SparseMatrixCSC, SubVector,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, findIJ, nnz,
    get_tril_to_full, transfer!, input_type, _madnlp_unsafe_wrap,
    is_supported, default_options

include(joinpath("..","deps","deps.jl"))

include("common.jl")
include("mc68.jl")

if @isdefined(libhsl)
    @isdefined(libma27) || const libma27 = libhsl
    @isdefined(libma57) || const libma57 = libhsl
    @isdefined(libma77) || const libma77 = libhsl
    @isdefined(libma86) || const libma86 = libhsl
    @isdefined(libma97) || const libma97 = libhsl
end

if @isdefined(libma27)
    include("ma27.jl")
    export Ma27Solver
end

if @isdefined(libma57)
    include("ma57.jl")
    export Ma57Solver
end

if @isdefined(libma77)
    include("ma77.jl")
    export Ma77Solver
end

if @isdefined(libma86)
    include("ma86.jl")
    export Ma86Solver
end

if @isdefined(libma97)
    include("ma97.jl")
    export Ma97Solver
end

function __init__()
    check_deps()
    try
        @isdefined(libhsl)  && dlopen(libhsl,RTLD_DEEPBIND)
        @isdefined(libma27) && dlopen(libma27,RTLD_DEEPBIND)
        @isdefined(libma77) && dlopen(libma57,RTLD_DEEPBIND)
        @isdefined(libma77) && dlopen(libma77,RTLD_DEEPBIND)
        @isdefined(libma86) && dlopen(libma77,RTLD_DEEPBIND)
        @isdefined(libma97) && dlopen(libma97,RTLD_DEEPBIND)
    catch e
        println("HSL shared library cannot be loaded")
    end
end


end # module
