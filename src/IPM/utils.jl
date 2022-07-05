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

MadNLPExecutionStats(ips::InteriorPointSolver) =MadNLPExecutionStats(
    ips.status,
    _madnlp_unsafe_wrap(ips.x, get_nvar(ips.nlp)),
    ips.obj_val,ips.c,
    ips.inf_du, ips.inf_pr,
    ips.l,
    _madnlp_unsafe_wrap(ips.zl, get_nvar(ips.nlp)),
    _madnlp_unsafe_wrap(ips.zu, get_nvar(ips.nlp)),
    ips.cnt.k, get_counters(ips.nlp),ips.cnt.total_time
)

get_counters(nlp::NLPModels.AbstractNLPModel) = nlp.counters
get_counters(nlp::NLPModels.AbstractNLSModel) = nlp.counters.counters
getStatus(result::MadNLPExecutionStats) = STATUS_OUTPUT_DICT[result.status]

# Utilities
has_constraints(ips) = ips.m != 0

# Print functions -----------------------------------------------------------
function print_init(ips::AbstractInteriorPointSolver)
    @notice(ips.logger,@sprintf("Number of nonzeros in constraint Jacobian............: %8i",get_nnzj(ips.nlp.meta)))
    @notice(ips.logger,@sprintf("Number of nonzeros in Lagrangian Hessian.............: %8i\n",get_nnzh(ips.nlp.meta)))

    num_fixed = length(ips.ind_fixed)
    num_var = get_nvar(ips.nlp) - num_fixed
    num_llb_vars = length(ips.ind_llb)
    num_lu_vars = sum((get_lvar(ips.nlp).!=-Inf).*(get_uvar(ips.nlp).!=Inf)) - num_fixed
    num_uub_vars = length(ips.ind_uub)
    num_eq_cons = sum(get_lcon(ips.nlp).==get_ucon(ips.nlp))
    num_ineq_cons = sum(get_lcon(ips.nlp).!=get_ucon(ips.nlp))
    num_ue_cons = sum((get_lcon(ips.nlp).!=get_ucon(ips.nlp)).*(get_lcon(ips.nlp).==-Inf).*(get_ucon(ips.nlp).!=Inf))
    num_le_cons = sum((get_lcon(ips.nlp).!=get_ucon(ips.nlp)).*(get_lcon(ips.nlp).!=-Inf).*(get_ucon(ips.nlp).==Inf))
    num_lu_cons = sum((get_lcon(ips.nlp).!=get_ucon(ips.nlp)).*(get_lcon(ips.nlp).!=-Inf).*(get_ucon(ips.nlp).!=Inf))
    get_nvar(ips.nlp) < num_eq_cons && throw(NotEnoughDegreesOfFreedomException())

    @notice(ips.logger,@sprintf("Total number of variables............................: %8i",num_var))
    @notice(ips.logger,@sprintf("                     variables with only lower bounds: %8i",num_llb_vars))
    @notice(ips.logger,@sprintf("                variables with lower and upper bounds: %8i",num_lu_vars))
    @notice(ips.logger,@sprintf("                     variables with only upper bounds: %8i",num_uub_vars))
    @notice(ips.logger,@sprintf("Total number of equality constraints.................: %8i",num_eq_cons))
    @notice(ips.logger,@sprintf("Total number of inequality constraints...............: %8i",num_ineq_cons))
    @notice(ips.logger,@sprintf("        inequality constraints with only lower bounds: %8i",num_le_cons))
    @notice(ips.logger,@sprintf("   inequality constraints with lower and upper bounds: %8i",num_lu_cons))
    @notice(ips.logger,@sprintf("        inequality constraints with only upper bounds: %8i\n",num_ue_cons))
    return
end

function print_iter(ips::AbstractInteriorPointSolver;is_resto=false)
    mod(ips.cnt.k,10)==0&& @info(ips.logger,@sprintf(
        "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls"))
    @info(ips.logger,@sprintf(
        "%4i%s% 10.7e %6.2e %6.2e %5.1f %6.2e %s %6.2e %6.2e%s  %i",
        ips.cnt.k,is_resto ? "r" : " ",ips.obj_val/ips.obj_scale[],
        is_resto ? ips.RR.inf_pr_R : ips.inf_pr,
        is_resto ? ips.RR.inf_du_R : ips.inf_du,
        is_resto ? log(10,ips.RR.mu_R) : log(10,ips.mu),
        ips.cnt.k == 0 ? 0. : norm(primal(ips.d),Inf),
        ips.del_w == 0 ? "   - " : @sprintf("%5.1f",log(10,ips.del_w)),
        ips.alpha_z,ips.alpha,ips.ftype,ips.cnt.l))
    return
end

function print_summary_1(ips::AbstractInteriorPointSolver)
    @notice(ips.logger,"")
    @notice(ips.logger,"Number of Iterations....: $(ips.cnt.k)\n")
    @notice(ips.logger,"                                   (scaled)                 (unscaled)")
    @notice(ips.logger,@sprintf("Objective...............:  % 1.16e   % 1.16e",ips.obj_val,ips.obj_val/ips.obj_scale[]))
    @notice(ips.logger,@sprintf("Dual infeasibility......:   %1.16e    %1.16e",ips.inf_du,ips.inf_du/ips.obj_scale[]))
    @notice(ips.logger,@sprintf("Constraint violation....:   %1.16e    %1.16e",norm(ips.c,Inf),ips.inf_pr))
    @notice(ips.logger,@sprintf("Complementarity.........:   %1.16e    %1.16e",
                                ips.inf_compl*ips.obj_scale[],ips.inf_compl))
    @notice(ips.logger,@sprintf("Overall NLP error.......:   %1.16e    %1.16e\n",
                                max(ips.inf_du*ips.obj_scale[],norm(ips.c,Inf),ips.inf_compl),
                                max(ips.inf_du,ips.inf_pr,ips.inf_compl)))
    return
end

function print_summary_2(ips::AbstractInteriorPointSolver)
    ips.cnt.solver_time = ips.cnt.total_time-ips.cnt.linear_solver_time-ips.cnt.eval_function_time
    @notice(ips.logger,"Number of objective function evaluations             = $(ips.cnt.obj_cnt)")
    @notice(ips.logger,"Number of objective gradient evaluations             = $(ips.cnt.obj_grad_cnt)")
    @notice(ips.logger,"Number of constraint evaluations                     = $(ips.cnt.con_cnt)")
    @notice(ips.logger,"Number of constraint Jacobian evaluations            = $(ips.cnt.con_jac_cnt)")
    @notice(ips.logger,"Number of Lagrangian Hessian evaluations             = $(ips.cnt.lag_hess_cnt)")
    @notice(ips.logger,@sprintf("Total wall-clock secs in solver (w/o fun. eval./lin. alg.)  = %6.3f",
                                ips.cnt.solver_time))
    @notice(ips.logger,@sprintf("Total wall-clock secs in linear solver                      = %6.3f",
                                ips.cnt.linear_solver_time))
    @notice(ips.logger,@sprintf("Total wall-clock secs in NLP function evaluations           = %6.3f",
                                ips.cnt.eval_function_time))
    @notice(ips.logger,@sprintf("Total wall-clock secs                                       = %6.3f\n",
                                ips.cnt.total_time))
end

function print_ignored_options(logger,option_dict)
    @warn(logger,"The following options are ignored: ")
    for (key,val) in option_dict
        @warn(logger," - "*string(key))
    end
end
function string(ips::AbstractInteriorPointSolver)
    """
                Interior point solver

                number of variables......................: $(get_nvar(ips.nlp))
                number of constraints....................: $(get_ncon(ips.nlp))
                number of nonzeros in lagrangian hessian.: $(get_nnzh(ips.nlp.meta))
                number of nonzeros in constraint jacobian: $(get_nnzj(ips.nlp.meta))
                status...................................: $(ips.status)
                """
end
print(io::IO,ips::AbstractInteriorPointSolver) = print(io, string(ips))
show(io::IO,ips::AbstractInteriorPointSolver) = print(io,ips)

