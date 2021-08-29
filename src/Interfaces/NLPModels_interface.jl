# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

const STATUS_TRANSLATOR = Dict(
    SOLVE_SUCCEEDED=>:first_order,
    SOLVED_TO_ACCEPTABLE_LEVEL=>:acceptable,
    INFEASIBLE_PROBLEM_DETECTED=>:infeasible,
    USER_REQUESTED_STOP=>:user,
    MAXIMUM_ITERATIONS_EXCEEDED=>:max_iter,
    MAXIMUM_WALLTIME_EXCEEDED=>:max_time)

function NonlinearProgram(model::AbstractNLPModel; buffered=true)
    buffered ? _nlp_model_to_nonlinear_program_buffered(model) : _nlp_model_to_nonlinear_program(model)
end

function _nlp_model_to_nonlinear_program(model)
    return NonlinearProgram(
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
        (I,J)->hess_structure!(model,I,J),(I,J)->jac_structure!(model,I,J),INITIAL,Dict{Symbol,Any}())
end

function _nlp_model_to_nonlinear_program_buffered(model)
    # buffers
    xb = Vector{Float64}(undef,model.meta.nvar)
    fb = Vector{Float64}(undef,model.meta.nvar)
    cb = Vector{Float64}(undef,model.meta.ncon)
    lb = cb
    jacb= Vector{Float64}(undef,model.meta.nnzj)
    hessb = Vector{Float64}(undef,model.meta.nnzh)

    return NonlinearProgram(
        model.meta.nvar,model.meta.ncon,model.meta.nnzh,model.meta.nnzj,
        0.,model.meta.x0,zeros(model.meta.ncon),model.meta.y0,
        zeros(model.meta.nvar),zeros(model.meta.nvar),
        model.meta.lvar,model.meta.uvar,model.meta.lcon,model.meta.ucon,
        model.meta.minimize ?
            (x)->obj(model,copyto!(xb,x)) : (x)->-obj(model,copyto!(xb,x)),
        model.meta.minimize ?
            (f,x)->(grad!(model,copyto!(xb,x),fb);copyto!(f,fb)) :
            (f,x)->(grad!(model,copyto!(xb,x),fb);copyto!(f,fb);f.*=-1.),
        (c,x)->(cons!(model,copyto!(xb,x),cb);copyto!(c,cb)),
        (jac,x)->(jac_coord!(model,copyto!(xb,x),jacb);copyto!(jac,jacb)),
        model.meta.minimize ?
            (hess,x,l,sig)->(hess_coord!(model,copyto!(xb,x),copyto!(lb,l),hessb;obj_weight= sig);
                             copyto!(hess,hessb)) :
            (hess,x,l,sig)->(hess_coord!(model,copyto!(xb,x),copyto!(lb,l),hessb;obj_weight= -sig);
                             copyto!(hess,hessb)),
        (I,J)->hess_structure!(model,I,J),(I,J)->jac_structure!(model,I,J),INITIAL,Dict{Symbol,Any}())
end


function madnlp(model::AbstractNLPModel;buffered=true, kwargs...)

    nlp = NonlinearProgram(model;buffered=buffered)
    ips = Solver(nlp;kwargs...)
    # seems that some CUTEst model requires this
    ips.f.=0
    ips.c.=0
    initialize!(ips.kkt)
    ips.zl.=0
    ips.zu.=0
    # ------------------------------------------
    optimize!(ips)

    return GenericExecutionStats(
        haskey(STATUS_TRANSLATOR,nlp.status) ? STATUS_TRANSLATOR[nlp.status] : :unknown,
        model,solution=nlp.x,
        objective=nlp.obj_val, dual_feas=ips.inf_du, iter=ips.cnt.k,
        primal_feas=ips.inf_pr, elapsed_time=ips.cnt.total_time, multipliers=nlp.l,
        multipliers_L=nlp.zl, multipliers_U=nlp.zu, solver_specific=Dict(:ips=>ips,:nlp=>nlp))
end
