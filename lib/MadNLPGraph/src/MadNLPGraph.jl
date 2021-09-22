module MadNLPGraph

import MadNLP:
    optimize!, Optimizer, InteriorPointSolver,
    MOIU, MOI, get_nnz_hess, get_nnz_jac, set_x!, set_g!, INITIAL, is_jac_hess_constant,
    jacobian_structure, hessian_lagrangian_structure, eval_objective,
    eval_objective_gradient, eval_function, eval_constraint, eval_hessian_lagrangian, eval_constraint_jacobian,
    @kwdef, Logger, @debug, @warn, @error, @sprintf,
    AbstractOptions, AbstractLinearSolver, EmptyLinearSolver, set_options!,
    SparseMatrixCSC, SubVector, StrideOneVector,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia,
    default_linear_solver, default_dense_solver, get_csc_view, get_cscsy_view, nnz, mul!,
    set_blas_num_threads, blas_num_threads, @blas_safe_threads,
    MOIModel, NLPModelsCounters, MadNLPExecutionStats, AbstractNLPModel, NLPModelMeta,
    get_lcon, get_ucon, jac_structure!, hess_structure!, obj, grad!, cons!, jac_coord!, hess_coord!
import Metis: partition
import LightGraphs: Graph, Edge, add_edge!, edges, src, dst, neighbors, nv
import Plasmo: OptiGraph, OptiNode, OptiEdge, all_nodes, all_edges, all_variables, num_all_nodes, getlinkconstraints, getnode, num_variables, num_constraints
import JuMP: _create_nlp_block_data, set_optimizer, GenericAffExpr

include("plasmo_interface.jl")
include("graphtools.jl")
include("schur.jl")
include("schwarz.jl")

export MadNLPSchur, MadNLPSchwarz

end # module
