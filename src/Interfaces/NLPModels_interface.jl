# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

NonlinearProgram(model::AbstractNLPModel) = NonlinearProgram(
    model.meta.nvar,model.meta.ncon,model.meta.nnzh,model.meta.nnzj,
    0.,model.meta.x0,zeros(model.meta.ncon),model.meta.y0,
    zeros(model.meta.nvar),zeros(model.meta.nvar),
    model.meta.lvar,model.meta.uvar,model.meta.lcon,model.meta.ucon,
    model.meta.minimize ? (x)->obj(model,x) : (x)->-obj(model,x),
    model.meta.minimize ? (f,x)->grad!(model,x,f) : (f,x)->(grad!(model,x,f);f.*=-1.),
    (c,x)->cons!(model,x,c),
    (jac,x)->jac_coord!(model,x,jac),
    model.meta.minimize ? (hess,x,l,sig)->hess_coord!(model,x,l,hess;obj_weight= sig) :
    (hess,x,l,sig)->hess_coord!(model,x,l,hess;obj_weight= -sig),
    (I,J)->hess_structure!(model,I,J),(I,J)->jac_structure!(model,I,J),:Initial,Dict{Symbol,Any}())

status_translator = Dict(
    :Solve_Succeeded=>:first_order,
    :Solved_To_Acceptable_Level=>:acceptable,
    :Infeasible_Problem_Detected=>:infeasible,
    :User_Requested_Stop=>:user,
    :Maximum_Iterations_Exceeded=>:max_iter,
    :Maximum_CpuTime_Exceeded=>:max_time)

function madnlp(model::AbstractNLPModel;kwargs...)
    
    nlp = NonlinearProgram(model)
    ips = Solver(nlp;kwargs...)
    # seems that some CUTEst model requires this
    ips.f.=0 
    ips.c.=0
    ips.jac.=0
    ips.hess.=0
    ips.zl.=0
    ips.zu.=0
    # ------------------------------------------
    optimize!(ips)
    finalize(model)

    return GenericExecutionStats(
        haskey(status_translator,nlp.status) ? status_translator[nlp.status] : :unknown,
        model,solution=nlp.x,
        objective=nlp.obj_val, dual_feas=ips.inf_du, iter=ips.cnt.k,
        primal_feas=ips.inf_pr, elapsed_time=ips.cnt.total_time, multipliers=nlp.l,
        multipliers_L=nlp.zl, multipliers_U=nlp.zu, solver_specific=Dict(:ips=>ips,:nlp=>nlp))
end
