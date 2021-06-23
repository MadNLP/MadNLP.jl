module MadNLPIterative

import MadNLP:
    @kwdef, Logger, @debug, @warn, @error,
    AbstractOptions, AbstractIterator, set_options!, @sprintf,
    solve_refine!, mul!, ldiv!, size, StrideOneVector
import IterativeSolvers:
    FastHessenberg, ArnoldiDecomp, Residual, init!, init_residual!, expand!, Identity,
    orthogonalize_and_normalize!, update_residual!, gmres_iterable!, GMRESIterable, converged,
    ModifiedGramSchmidt

include("krylov.jl")

export MadNLPKrylov

end # module
