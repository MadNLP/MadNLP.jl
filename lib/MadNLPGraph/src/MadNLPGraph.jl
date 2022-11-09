module MadNLPGraph

import MadNLP:
    solve!, Optimizer, MadNLPSolver,
    MOIU, MOI, INITIAL,
    @kwdef, MadNLPLogger, @debug, @warn, @error, @sprintf,
    AbstractOptions, AbstractLinearSolver, set_options!,
    SparseMatrixCSC, SubVector,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia,
    default_linear_solver, default_dense_solver, get_csc_view, get_cscsy_view, nnz, mul!,
    set_blas_num_threads, blas_num_threads, @blas_safe_threads,
    MOIModel, MadNLPExecutionStats, AbstractNLPModel, NLPModelMeta,
    get_lcon, get_ucon, jac_structure!, hess_structure!, obj, grad!, cons!, jac_coord!, hess_coord!
import MadNLP
import Metis: partition
import LightGraphs: Graph, Edge, add_edge!, edges, src, dst, neighbors, nv
import Plasmo: OptiGraph, OptiNode, OptiEdge, all_nodes, all_edges, all_variables, num_all_nodes, link_constraints, getnode, num_variables, num_constraints
import JuMP: set_optimizer, GenericAffExpr
import Plasmo
import JuMP

include("plasmo_interface.jl")
include("graphtools.jl")
include("schur.jl")
include("schwarz.jl")

export MadNLPSchur, MadNLPSchwarz

end # module
