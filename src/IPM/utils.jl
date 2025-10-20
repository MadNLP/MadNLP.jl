"""
    MadNLPExecutionStats{T, VT} <: AbstractExecutionStats

Store the results returned by MadNLP once the interior-point
algorithm has terminated.

"""
mutable struct MadNLPExecutionStats{T, VT} <: AbstractExecutionStats
    options::AbstractOptions
    status::Status
    solution::VT
    objective::T
    constraints::VT
    dual_feas::T
    primal_feas::T
    multipliers::VT
    multipliers_L::VT
    multipliers_U::VT
    iter::Int
    counters::MadNLPCounters
end

MadNLPExecutionStats(solver::AbstractMadNLPSolver) =MadNLPExecutionStats(
    get_opt(solver),
    get_status(solver),
    primal(get_x(solver))[1:get_nvar(get_nlp(solver))],
    get_obj_val(solver) / get_cb(solver).obj_scale[],
    get_c(solver) ./ get_cb(solver).con_scale,
    get_inf_du(solver),
    get_inf_pr(solver),
    copy(get_y(solver)),
    primal(get_zl(solver))[1:get_nvar(get_nlp(solver))],
    primal(get_zu(solver))[1:get_nvar(get_nlp(solver))],
    0,
    get_cnt(solver),
)

function update!(stats::MadNLPExecutionStats, solver::AbstractMadNLPSolver)
    stats.status = get_status(solver)
    stats.solution .= @view(primal(get_x(solver))[1:get_nvar(get_nlp(solver))])
    stats.multipliers .= (get_y(solver) .* get_cb(solver).con_scale) ./ get_cb(solver).obj_scale[]
    stats.multipliers_L .= @view(primal(get_zl(solver))[1:get_nvar(get_nlp(solver))]) ./ get_cb(solver).obj_scale[]
    stats.multipliers_U .= @view(primal(get_zu(solver))[1:get_nvar(get_nlp(solver))]) ./ get_cb(solver).obj_scale[]
    # stats.solution .= min.(
    #     max.(
    #         @view(primal(get_x(solver))[1:get_nvar(get_nlp(solver))]),
    #         get_lvar(get_nlp(solver))
    #     ),
    #     get_uvar(get_nlp(solver))
    # )
    stats.objective = get_obj_val(solver) / get_cb(solver).obj_scale[]
    stats.constraints .= get_c(solver) ./ get_cb(solver).con_scale .+ get_rhs(solver)
    stats.constraints[get_ind_ineq(solver)] .+= slack(get_x(solver))
    stats.dual_feas = get_inf_du(solver)
    stats.primal_feas = get_inf_pr(solver)
    update_z!(get_cb(solver), stats.multipliers_L, stats.multipliers_U, get_jacl(solver))
    stats.iter = get_cnt(solver).k
    return stats
end

get_counters(nlp::NLPModels.AbstractNLPModel) = nlp.counters
get_counters(nlp::NLPModels.AbstractNLSModel) = nlp.counters.counters
getStatus(result::MadNLPExecutionStats) = get_status_output(result.status, result.options)

# Exceptions
struct InvalidNumberException <: Exception
    callback::Symbol
end
struct NotEnoughDegreesOfFreedomException <: Exception end

# Utilities
has_constraints(solver) = get_m(solver) != 0

function get_vars_info(solver)
    nlp = get_nlp(solver)

    x_lb = get_lvar(nlp)
    x_ub = get_uvar(nlp)
    num_fixed = length(get_ind_fixed(solver))
    num_var = get_nvar(nlp) - num_fixed
    num_llb_vars = length(get_ind_llb(solver))

    # TODO make this non-allocating
    num_lu_vars = sum((x_lb .!=-Inf) .& (x_ub .!= Inf)) - num_fixed
    num_uub_vars = length(get_ind_uub(solver))
    return (
        n_free=num_var,
        n_fixed=num_fixed,
        n_only_lb=num_llb_vars,
        n_only_ub=num_uub_vars,
        n_bounded=num_lu_vars,
    )
end

function get_cons_info(solver)
    nlp = get_nlp(solver)

    g_lb = get_lcon(nlp)
    g_ub = get_ucon(nlp)

    # TODO make this non-allocating
    num_eq_cons = sum(g_lb .== g_ub)
    num_ineq_cons = length(g_lb) - num_eq_cons
    num_le_cons = sum((g_lb .!= -Inf) .& (g_ub .==  Inf))
    num_ue_cons = sum((g_ub .!=  Inf) .& (g_lb .== -Inf))
    num_lu_cons = num_ineq_cons - num_le_cons - num_ue_cons

    return (
        n_eq=num_eq_cons,
        n_ineq=num_ineq_cons,
        n_only_lb=num_le_cons,
        n_only_ub=num_ue_cons,
        n_bounded=num_lu_cons,
    )
end

# Print functions -----------------------------------------------------------
function print_init(solver::AbstractMadNLPSolver)
    @notice(get_logger(solver),@sprintf("Number of nonzeros in constraint Jacobian............: %8i", get_nnzj(get_nlp(solver).meta)))
    @notice(get_logger(solver),@sprintf("Number of nonzeros in Lagrangian Hessian.............: %8i\n", get_nnzh(get_nlp(solver).meta)))
    var_info = get_vars_info(solver)
    con_info = get_cons_info(solver)

    if get_nvar(get_nlp(solver)) < con_info.n_eq
        throw(NotEnoughDegreesOfFreedomException())
    end

    @notice(get_logger(solver),@sprintf("Total number of variables............................: %8i",var_info.n_free))
    @notice(get_logger(solver),@sprintf("                     variables with only lower bounds: %8i",var_info.n_only_lb))
    @notice(get_logger(solver),@sprintf("                variables with lower and upper bounds: %8i",var_info.n_bounded))
    @notice(get_logger(solver),@sprintf("                     variables with only upper bounds: %8i",var_info.n_only_ub))
    @notice(get_logger(solver),@sprintf("Total number of equality constraints.................: %8i",con_info.n_eq))
    @notice(get_logger(solver),@sprintf("Total number of inequality constraints...............: %8i",con_info.n_ineq))
    @notice(get_logger(solver),@sprintf("        inequality constraints with only lower bounds: %8i",con_info.n_only_lb))
    @notice(get_logger(solver),@sprintf("   inequality constraints with lower and upper bounds: %8i",con_info.n_bounded))
    @notice(get_logger(solver),@sprintf("        inequality constraints with only upper bounds: %8i\n",con_info.n_only_ub))
    return
end

function print_iter(solver::AbstractMadNLPSolver; is_resto=false)
    obj_scale = get_cb(solver).obj_scale[]
    mod(get_cnt(solver).k,10)==0&& @info(get_logger(solver),@sprintf(
        "iter    objective    inf_pr   inf_du inf_compl lg(mu) lg(rg) alpha_pr ir ls"))
    if is_resto
        RR = get_RR(solver)::RobustRestorer
        inf_du = RR.inf_du_R
        inf_pr = RR.inf_pr_R
        inf_compl = RR.inf_compl_R
        mu = log10(RR.mu_R)
    else
        inf_du = get_inf_du(solver)
        inf_pr = get_inf_pr(solver)
        inf_compl = get_inf_compl(solver)
        mu = log10(get_mu(solver))
    end
    @info(get_logger(solver),@sprintf(
        "%4i%s% 10.7e %6.2e %6.2e %7.2e %5.1f  %s  %6.2e %2i %2i%s",
        get_cnt(solver).k,is_resto ? "r" : " ",get_obj_val(solver)/obj_scale,
        inf_pr, inf_du, inf_compl, mu,
        # get_cnt(solver).k == 0 ? 0. : norm(primal(get_d(solver)),Inf),
        get_del_w(solver) == 0 ? "   - " : @sprintf("%5.1f",log(10,get_del_w(solver))),
        get_alpha(solver),
        get_cnt(solver).ir,
        get_cnt(solver).l,
        get_ftype(solver),))
    return
end

function print_summary(solver::AbstractMadNLPSolver)
    # TODO inquire this from nlpmodel wrapper
    obj_scale = get_cb(solver).obj_scale[]
    get_cnt(solver).solver_time = get_cnt(solver).total_time-get_cnt(solver).linear_solver_time-get_cnt(solver).eval_function_time

    @notice(get_logger(solver),"")
    @notice(get_logger(solver),"Number of Iterations....: $(get_cnt(solver).k)\n")
    @notice(get_logger(solver),"                                   (scaled)                 (unscaled)")
    @notice(get_logger(solver),@sprintf("Objective...............:  % 1.16e   % 1.16e",get_obj_val(solver),get_obj_val(solver)/obj_scale))
    @notice(get_logger(solver),@sprintf("Dual infeasibility......:   %1.16e    %1.16e",get_inf_du(solver),get_inf_du(solver)/obj_scale))
    @notice(get_logger(solver),@sprintf("Constraint violation....:   %1.16e    %1.16e",norm(get_c(solver),Inf),get_inf_pr(solver)))
    @notice(get_logger(solver),@sprintf("Complementarity.........:   %1.16e    %1.16e",
                                get_inf_compl(solver)*obj_scale,get_inf_compl(solver)))
    @notice(get_logger(solver),@sprintf("Overall NLP error.......:   %1.16e    %1.16e\n",
                                max(get_inf_du(solver)*obj_scale,norm(get_c(solver),Inf),get_inf_compl(solver)),
                                max(get_inf_du(solver),get_inf_pr(solver),get_inf_compl(solver))))

    @notice(get_logger(solver),"Number of objective function evaluations              = $(get_cnt(solver).obj_cnt)")
    @notice(get_logger(solver),"Number of objective gradient evaluations              = $(get_cnt(solver).obj_grad_cnt)")
    @notice(get_logger(solver),"Number of constraint evaluations                      = $(get_cnt(solver).con_cnt)")
    @notice(get_logger(solver),"Number of constraint Jacobian evaluations             = $(get_cnt(solver).con_jac_cnt)")
    @notice(get_logger(solver),"Number of Lagrangian Hessian evaluations              = $(get_cnt(solver).lag_hess_cnt)\n")
    @notice(get_logger(solver),@sprintf("Total wall secs in initialization                     = %6.3f",
                                get_cnt(solver).init_time))
    @notice(get_logger(solver),@sprintf("Total wall secs in linear solver                      = %6.3f",
                                get_cnt(solver).linear_solver_time))
    @notice(get_logger(solver),@sprintf("Total wall secs in NLP function evaluations           = %6.3f",
                                get_cnt(solver).eval_function_time))
    @notice(get_logger(solver),@sprintf("Total wall secs in solver (w/o init./fun./lin. alg.)  = %6.3f",
                                get_cnt(solver).total_time - get_cnt(solver).init_time - get_cnt(solver).linear_solver_time - get_cnt(solver).eval_function_time))
    @notice(get_logger(solver),@sprintf("Total wall secs                                       = %6.3f\n",
                                get_cnt(solver).total_time))
end


function string(solver::AbstractMadNLPSolver)
    """
                Interior point solver

                number of variables......................: $(get_nvar(get_nlp(solver)))
                number of constraints....................: $(get_ncon(get_nlp(solver)))
                number of nonzeros in Lagrangian Hessian.: $(get_nnzh(get_nlp(solver).meta))
                number of nonzeros in constraint Jacobian: $(get_nnzj(get_nlp(solver).meta))
                status...................................: $(get_status(solver))
                """
end
print(io::IO,solver::AbstractMadNLPSolver) = print(io, string(solver))
show(io::IO,solver::AbstractMadNLPSolver) = print(io,solver)
