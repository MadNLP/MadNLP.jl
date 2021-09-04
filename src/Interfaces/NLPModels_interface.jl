# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

const STATUS_TRANSLATOR = Dict(
    SOLVE_SUCCEEDED=>:first_order,
    SOLVED_TO_ACCEPTABLE_LEVEL=>:acceptable,
    INFEASIBLE_PROBLEM_DETECTED=>:infeasible,
    USER_REQUESTED_STOP=>:user,
    MAXIMUM_ITERATIONS_EXCEEDED=>:max_iter,
    MAXIMUM_WALLTIME_EXCEEDED=>:max_time)

function madnlp(model::AbstractNLPModel;buffered=true, kwargs...)
    ips = Solver(model;kwargs...)
    initialize!(ips.kkt)
    optimize!(ips)
    return GenericExecutionStats(
        haskey(STATUS_TRANSLATOR,ips.status) ? STATUS_TRANSLATOR[ips.status] : :unknown,
        ips.nlp,solution=view(ips.x,1:get_nvar(ips.nlp)),
        objective=ips.obj_val/ips.obj_scale[], dual_feas=ips.inf_du, iter=ips.cnt.k,
        primal_feas=ips.inf_pr, elapsed_time=ips.cnt.total_time, multipliers=ips.l,
        multipliers_L=view(ips.zl,1:get_nvar(ips.nlp)),
        multipliers_U=view(ips.zu,1:get_nvar(ips.nlp)),
        solver_specific=Dict(:madnlp_solver=>ips))
end

"Dense Jacobian callback"
function jac_dense! end

"Dense Hessian callback"
function hess_dense! end
