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
    _opt(solver),
    _status(solver),
    primal(_x(solver))[1:get_nvar(_nlp(solver))],
    _obj_val(solver) / _cb(solver).obj_scale[],
    _c(solver) ./ _cb(solver).con_scale,
    _inf_du(solver),
    _inf_pr(solver),
    copy(_y(solver)),
    primal(_zl(solver))[1:get_nvar(_nlp(solver))],
    primal(_zu(solver))[1:get_nvar(_nlp(solver))],
    0,
    _cnt(solver),
)

function update!(stats::MadNLPExecutionStats, solver::AbstractMadNLPSolver)
    stats.status = _status(solver)
    stats.solution .= @view(primal(_x(solver))[1:get_nvar(_nlp(solver))])
    stats.multipliers .= (_y(solver) .* _cb(solver).con_scale) ./ _cb(solver).obj_scale[]
    stats.multipliers_L .= @view(primal(_zl(solver))[1:get_nvar(_nlp(solver))]) ./ _cb(solver).obj_scale[]
    stats.multipliers_U .= @view(primal(_zu(solver))[1:get_nvar(_nlp(solver))]) ./ _cb(solver).obj_scale[]
    # stats.solution .= min.(
    #     max.(
    #         @view(primal(_x(solver))[1:get_nvar(_nlp(solver))]),
    #         get_lvar(_nlp(solver))
    #     ),
    #     get_uvar(_nlp(solver))
    # )
    stats.objective = _obj_val(solver) / _cb(solver).obj_scale[]
    stats.constraints .= _c(solver) ./ _cb(solver).con_scale .+ _rhs(solver)
    stats.constraints[_ind_ineq(solver)] .+= slack(_x(solver))
    stats.dual_feas = _inf_du(solver)
    stats.primal_feas = _inf_pr(solver)
    update_z!(_cb(solver), stats.multipliers_L, stats.multipliers_U, _jacl(solver))
    stats.iter = _cnt(solver).k
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
has_constraints(solver) = _m(solver) != 0

function get_vars_info(solver)
    nlp = _nlp(solver)

    x_lb = get_lvar(nlp)
    x_ub = get_uvar(nlp)
    num_fixed = length(_ind_fixed(solver))
    num_var = get_nvar(nlp) - num_fixed
    num_llb_vars = length(_ind_llb(solver))

    # TODO make this non-allocating
    num_lu_vars = sum((x_lb .!=-Inf) .& (x_ub .!= Inf)) - num_fixed
    num_uub_vars = length(_ind_uub(solver))
    return (
        n_free=num_var,
        n_fixed=num_fixed,
        n_only_lb=num_llb_vars,
        n_only_ub=num_uub_vars,
        n_bounded=num_lu_vars,
    )
end

function get_cons_info(solver)
    nlp = _nlp(solver)

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
    @notice(_logger(solver),@sprintf("Number of nonzeros in constraint Jacobian............: %8i", get_nnzj(_nlp(solver).meta)))
    @notice(_logger(solver),@sprintf("Number of nonzeros in Lagrangian Hessian.............: %8i\n", get_nnzh(_nlp(solver).meta)))
    var_info = get_vars_info(solver)
    con_info = get_cons_info(solver)

    if get_nvar(_nlp(solver)) < con_info.n_eq
        throw(NotEnoughDegreesOfFreedomException())
    end

    @notice(_logger(solver),@sprintf("Total number of variables............................: %8i",var_info.n_free))
    @notice(_logger(solver),@sprintf("                     variables with only lower bounds: %8i",var_info.n_only_lb))
    @notice(_logger(solver),@sprintf("                variables with lower and upper bounds: %8i",var_info.n_bounded))
    @notice(_logger(solver),@sprintf("                     variables with only upper bounds: %8i",var_info.n_only_ub))
    @notice(_logger(solver),@sprintf("Total number of equality constraints.................: %8i",con_info.n_eq))
    @notice(_logger(solver),@sprintf("Total number of inequality constraints...............: %8i",con_info.n_ineq))
    @notice(_logger(solver),@sprintf("        inequality constraints with only lower bounds: %8i",con_info.n_only_lb))
    @notice(_logger(solver),@sprintf("   inequality constraints with lower and upper bounds: %8i",con_info.n_bounded))
    @notice(_logger(solver),@sprintf("        inequality constraints with only upper bounds: %8i\n",con_info.n_only_ub))
    return
end

function print_iter(solver::AbstractMadNLPSolver; is_resto=false)
    obj_scale = _cb(solver).obj_scale[]
    mod(_cnt(solver).k,10)==0&& @info(_logger(solver),@sprintf(
        "iter    objective    inf_pr   inf_du inf_compl lg(mu) lg(rg) alpha_pr ir ls"))
    if is_resto
        RR = _RR(solver)::RobustRestorer
        inf_du = RR.inf_du_R
        inf_pr = RR.inf_pr_R
        inf_compl = RR.inf_compl_R
        mu = log10(RR.mu_R)
    else
        inf_du = _inf_du(solver)
        inf_pr = _inf_pr(solver)
        inf_compl = _inf_compl(solver)
        mu = log10(_mu(solver))
    end
    @info(_logger(solver),@sprintf(
        "%4i%s% 10.7e %6.2e %6.2e %7.2e %5.1f  %s  %6.2e %2i %2i%s",
        _cnt(solver).k,is_resto ? "r" : " ",_obj_val(solver)/obj_scale,
        inf_pr, inf_du, inf_compl, mu,
        # _cnt(solver).k == 0 ? 0. : norm(primal(_d(solver)),Inf),
        _del_w(solver) == 0 ? "   - " : @sprintf("%5.1f",log(10,_del_w(solver))),
        _alpha(solver),
        _cnt(solver).ir,
        _cnt(solver).l,
        _ftype(solver),))
    return
end

function print_summary(solver::AbstractMadNLPSolver)
    # TODO inquire this from nlpmodel wrapper
    obj_scale = _cb(solver).obj_scale[]
    _cnt(solver).solver_time = _cnt(solver).total_time-_cnt(solver).linear_solver_time-_cnt(solver).eval_function_time

    @notice(_logger(solver),"")
    @notice(_logger(solver),"Number of Iterations....: $(_cnt(solver).k)\n")
    @notice(_logger(solver),"                                   (scaled)                 (unscaled)")
    @notice(_logger(solver),@sprintf("Objective...............:  % 1.16e   % 1.16e",_obj_val(solver),_obj_val(solver)/obj_scale))
    @notice(_logger(solver),@sprintf("Dual infeasibility......:   %1.16e    %1.16e",_inf_du(solver),_inf_du(solver)/obj_scale))
    @notice(_logger(solver),@sprintf("Constraint violation....:   %1.16e    %1.16e",norm(_c(solver),Inf),_inf_pr(solver)))
    @notice(_logger(solver),@sprintf("Complementarity.........:   %1.16e    %1.16e",
                                _inf_compl(solver)*obj_scale,_inf_compl(solver)))
    @notice(_logger(solver),@sprintf("Overall NLP error.......:   %1.16e    %1.16e\n",
                                max(_inf_du(solver)*obj_scale,norm(_c(solver),Inf),_inf_compl(solver)),
                                max(_inf_du(solver),_inf_pr(solver),_inf_compl(solver))))

    @notice(_logger(solver),"Number of objective function evaluations              = $(_cnt(solver).obj_cnt)")
    @notice(_logger(solver),"Number of objective gradient evaluations              = $(_cnt(solver).obj_grad_cnt)")
    @notice(_logger(solver),"Number of constraint evaluations                      = $(_cnt(solver).con_cnt)")
    @notice(_logger(solver),"Number of constraint Jacobian evaluations             = $(_cnt(solver).con_jac_cnt)")
    @notice(_logger(solver),"Number of Lagrangian Hessian evaluations              = $(_cnt(solver).lag_hess_cnt)\n")
    @notice(_logger(solver),@sprintf("Total wall secs in initialization                     = %6.3f",
                                _cnt(solver).init_time))
    @notice(_logger(solver),@sprintf("Total wall secs in linear solver                      = %6.3f",
                                _cnt(solver).linear_solver_time))
    @notice(_logger(solver),@sprintf("Total wall secs in NLP function evaluations           = %6.3f",
                                _cnt(solver).eval_function_time))
    @notice(_logger(solver),@sprintf("Total wall secs in solver (w/o init./fun./lin. alg.)  = %6.3f",
                                _cnt(solver).total_time - _cnt(solver).init_time - _cnt(solver).linear_solver_time - _cnt(solver).eval_function_time))
    @notice(_logger(solver),@sprintf("Total wall secs                                       = %6.3f\n",
                                _cnt(solver).total_time))
end


function string(solver::AbstractMadNLPSolver)
    """
                Interior point solver

                number of variables......................: $(get_nvar(_nlp(solver)))
                number of constraints....................: $(get_ncon(_nlp(solver)))
                number of nonzeros in Lagrangian Hessian.: $(get_nnzh(_nlp(solver).meta))
                number of nonzeros in constraint Jacobian: $(get_nnzj(_nlp(solver).meta))
                status...................................: $(_status(solver))
                """
end
print(io::IO,solver::AbstractMadNLPSolver) = print(io, string(solver))
show(io::IO,solver::AbstractMadNLPSolver) = print(io,solver)
