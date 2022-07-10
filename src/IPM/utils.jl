struct MadNLPExecutionStats{T} <: AbstractExecutionStats
    status::Status
    solution::Vector{T}
    objective::T
    constraints::Vector{T}
    dual_feas::T
    primal_feas::T
    multipliers::Vector{T}
    multipliers_L::Vector{T}
    multipliers_U::Vector{T}
    iter::Int
    counters::NLPModels.Counters
    elapsed_time::Real
end

struct InvalidNumberException <: Exception end
struct NotEnoughDegreesOfFreedomException <: Exception end

MadNLPExecutionStats(solver::MadNLPSolver) =MadNLPExecutionStats(
    solver.status,
    _madnlp_unsafe_wrap(solver.x, get_nvar(solver.nlp)),
    solver.obj_val,solver.c,
    solver.inf_du, solver.inf_pr,
    solver.y,
    _madnlp_unsafe_wrap(solver.zl, get_nvar(solver.nlp)),
    _madnlp_unsafe_wrap(solver.zu, get_nvar(solver.nlp)),
    solver.cnt.k, get_counters(solver.nlp),solver.cnt.total_time
)

get_counters(nlp::NLPModels.AbstractNLPModel) = nlp.counters
get_counters(nlp::NLPModels.AbstractNLSModel) = nlp.counters.counters
getStatus(result::MadNLPExecutionStats) = STATUS_OUTPUT_DICT[result.status]

# Utilities
has_constraints(solver) = solver.m != 0

# Print functions -----------------------------------------------------------
function print_init(solver::AbstractMadNLPSolver)
    @notice(solver.logger,@sprintf("Number of nonzeros in constraint Jacobian............: %8i",get_nnzj(solver.nlp.meta)))
    @notice(solver.logger,@sprintf("Number of nonzeros in Lagrangian Hessian.............: %8i\n",get_nnzh(solver.nlp.meta)))

    num_fixed = length(solver.ind_fixed)
    num_var = get_nvar(solver.nlp) - num_fixed
    num_llb_vars = length(solver.ind_llb)
    num_lu_vars = sum((get_lvar(solver.nlp).!=-Inf).*(get_uvar(solver.nlp).!=Inf)) - num_fixed
    num_uub_vars = length(solver.ind_uub)
    num_eq_cons = sum(get_lcon(solver.nlp).==get_ucon(solver.nlp))
    num_ineq_cons = sum(get_lcon(solver.nlp).!=get_ucon(solver.nlp))
    num_ue_cons = sum((get_lcon(solver.nlp).!=get_ucon(solver.nlp)).*(get_lcon(solver.nlp).==-Inf).*(get_ucon(solver.nlp).!=Inf))
    num_le_cons = sum((get_lcon(solver.nlp).!=get_ucon(solver.nlp)).*(get_lcon(solver.nlp).!=-Inf).*(get_ucon(solver.nlp).==Inf))
    num_lu_cons = sum((get_lcon(solver.nlp).!=get_ucon(solver.nlp)).*(get_lcon(solver.nlp).!=-Inf).*(get_ucon(solver.nlp).!=Inf))
    get_nvar(solver.nlp) < num_eq_cons && throw(NotEnoughDegreesOfFreedomException())

    @notice(solver.logger,@sprintf("Total number of variables............................: %8i",num_var))
    @notice(solver.logger,@sprintf("                     variables with only lower bounds: %8i",num_llb_vars))
    @notice(solver.logger,@sprintf("                variables with lower and upper bounds: %8i",num_lu_vars))
    @notice(solver.logger,@sprintf("                     variables with only upper bounds: %8i",num_uub_vars))
    @notice(solver.logger,@sprintf("Total number of equality constraints.................: %8i",num_eq_cons))
    @notice(solver.logger,@sprintf("Total number of inequality constraints...............: %8i",num_ineq_cons))
    @notice(solver.logger,@sprintf("        inequality constraints with only lower bounds: %8i",num_le_cons))
    @notice(solver.logger,@sprintf("   inequality constraints with lower and upper bounds: %8i",num_lu_cons))
    @notice(solver.logger,@sprintf("        inequality constraints with only upper bounds: %8i\n",num_ue_cons))
    return
end

function print_iter(solver::AbstractMadNLPSolver;is_resto=false)
    mod(solver.cnt.k,10)==0&& @info(solver.logger,@sprintf(
        "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls"))
    @info(solver.logger,@sprintf(
        "%4i%s% 10.7e %6.2e %6.2e %5.1f %6.2e %s %6.2e %6.2e%s  %i",
        solver.cnt.k,is_resto ? "r" : " ",solver.obj_val/solver.obj_scale[],
        is_resto ? solver.RR.inf_pr_R : solver.inf_pr,
        is_resto ? solver.RR.inf_du_R : solver.inf_du,
        is_resto ? log(10,solver.RR.mu_R) : log(10,solver.mu),
        solver.cnt.k == 0 ? 0. : norm(primal(solver.d),Inf),
        solver.del_w == 0 ? "   - " : @sprintf("%5.1f",log(10,solver.del_w)),
        solver.alpha_z,solver.alpha,solver.ftype,solver.cnt.l))
    return
end

function print_summary_1(solver::AbstractMadNLPSolver)
    @notice(solver.logger,"")
    @notice(solver.logger,"Number of Iterations....: $(solver.cnt.k)\n")
    @notice(solver.logger,"                                   (scaled)                 (unscaled)")
    @notice(solver.logger,@sprintf("Objective...............:  % 1.16e   % 1.16e",solver.obj_val,solver.obj_val/solver.obj_scale[]))
    @notice(solver.logger,@sprintf("Dual infeasibility......:   %1.16e    %1.16e",solver.inf_du,solver.inf_du/solver.obj_scale[]))
    @notice(solver.logger,@sprintf("Constraint violation....:   %1.16e    %1.16e",norm(solver.c,Inf),solver.inf_pr))
    @notice(solver.logger,@sprintf("Complementarity.........:   %1.16e    %1.16e",
                                solver.inf_compl*solver.obj_scale[],solver.inf_compl))
    @notice(solver.logger,@sprintf("Overall NLP error.......:   %1.16e    %1.16e\n",
                                max(solver.inf_du*solver.obj_scale[],norm(solver.c,Inf),solver.inf_compl),
                                max(solver.inf_du,solver.inf_pr,solver.inf_compl)))
    return
end

function print_summary_2(solver::AbstractMadNLPSolver)
    solver.cnt.solver_time = solver.cnt.total_time-solver.cnt.linear_solver_time-solver.cnt.eval_function_time
    @notice(solver.logger,"Number of objective function evaluations             = $(solver.cnt.obj_cnt)")
    @notice(solver.logger,"Number of objective gradient evaluations             = $(solver.cnt.obj_grad_cnt)")
    @notice(solver.logger,"Number of constraint evaluations                     = $(solver.cnt.con_cnt)")
    @notice(solver.logger,"Number of constraint Jacobian evaluations            = $(solver.cnt.con_jac_cnt)")
    @notice(solver.logger,"Number of Lagrangian Hessian evaluations             = $(solver.cnt.lag_hess_cnt)")
    @notice(solver.logger,@sprintf("Total wall-clock secs in solver (w/o fun. eval./lin. alg.)  = %6.3f",
                                solver.cnt.solver_time))
    @notice(solver.logger,@sprintf("Total wall-clock secs in linear solver                      = %6.3f",
                                solver.cnt.linear_solver_time))
    @notice(solver.logger,@sprintf("Total wall-clock secs in NLP function evaluations           = %6.3f",
                                solver.cnt.eval_function_time))
    @notice(solver.logger,@sprintf("Total wall-clock secs                                       = %6.3f\n",
                                solver.cnt.total_time))
end

function print_ignored_options(logger,option_dict)
    @warn(logger,"The following options are ignored: ")
    for (key,val) in option_dict
        @warn(logger," - "*string(key))
    end
end
function string(solver::AbstractMadNLPSolver)
    """
                Interior point solver

                number of variables......................: $(get_nvar(solver.nlp))
                number of constraints....................: $(get_ncon(solver.nlp))
                number of nonzeros in lagrangian hessian.: $(get_nnzh(solver.nlp.meta))
                number of nonzeros in constraint jacobian: $(get_nnzj(solver.nlp.meta))
                status...................................: $(solver.status)
                """
end
print(io::IO,solver::AbstractMadNLPSolver) = print(io, string(solver))
show(io::IO,solver::AbstractMadNLPSolver) = print(io,solver)

