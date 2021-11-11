# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

@kwdef mutable struct Counters
    k::Int = 0 # total iteration counter
    l::Int = 0 # backtracking line search counter
    t::Int = 0 # restoration phase counter

    start_time::Float64

    linear_solver_time::Float64 = 0.
    eval_function_time::Float64 = 0.
    solver_time::Float64 = 0.
    total_time::Float64 = 0.

    obj_cnt::Int = 0
    obj_grad_cnt::Int = 0
    con_cnt::Int = 0
    con_jac_cnt::Int = 0
    lag_hess_cnt::Int = 0

    acceptable_cnt::Int = 0
end

mutable struct RobustRestorer
    obj_val_R::Float64
    f_R::Vector{Float64}
    x_ref::Vector{Float64}

    theta_ref::Float64
    D_R::Vector{Float64}
    obj_val_R_trial::Float64

    pp::Vector{Float64}
    nn::Vector{Float64}
    zp::Vector{Float64}
    zn::Vector{Float64}

    dpp::Vector{Float64}
    dnn::Vector{Float64}
    dzp::Vector{Float64}
    dzn::Vector{Float64}

    pp_trial::Vector{Float64}
    nn_trial::Vector{Float64}

    inf_pr_R::Float64
    inf_du_R::Float64
    inf_compl_R::Float64

    mu_R::Float64
    tau_R::Float64
    zeta::Float64

    filter::Vector{Tuple{Float64,Float64}}
end

abstract type AbstractInteriorPointSolver end

mutable struct InteriorPointSolver{KKTSystem} <: AbstractInteriorPointSolver
    nlp::AbstractNLPModel
    kkt::KKTSystem

    opt::Options
    cnt::Counters
    logger::Logger

    n::Int # number of variables (after reformulation)
    m::Int # number of cons
    nlb::Int
    nub::Int

    x::Vector{Float64} # primal (after reformulation)
    l::Vector{Float64} # dual
    zl::Vector{Float64} # dual (after reformulation)
    zu::Vector{Float64} # dual (after reformulation)
    xl::Vector{Float64} # primal lower bound (after reformulation)
    xu::Vector{Float64} # primal upper bound (after reformulation)

    obj_val::Float64
    f::Vector{Float64}
    c::Vector{Float64}

    jacl::Vector{Float64}

    d::Vector{Float64}
    dx::StrideOneVector{Float64}
    dl::StrideOneVector{Float64}
    dzl::StrideOneVector{Float64}
    dzu::StrideOneVector{Float64}

    p::Vector{Float64}
    px::StrideOneVector{Float64}
    pl::StrideOneVector{Float64}

    pzl::Union{Nothing,StrideOneVector{Float64}}
    pzu::Union{Nothing,StrideOneVector{Float64}}

    _w1::Vector{Float64}
    _w1x::StrideOneVector{Float64}
    _w1l::StrideOneVector{Float64}
    _w1zl::Union{Nothing,StrideOneVector{Float64}}
    _w1zu::Union{Nothing,StrideOneVector{Float64}}

    _w2::Vector{Float64}
    _w2x::StrideOneVector{Float64}
    _w2l::StrideOneVector{Float64}
    _w2zl::Union{Nothing,StrideOneVector{Float64}}
    _w2zu::Union{Nothing,StrideOneVector{Float64}}

    _w3::Vector{Float64}
    _w3x::StrideOneVector{Float64}
    _w3l::StrideOneVector{Float64}

    _w4::Vector{Float64}
    _w4x::StrideOneVector{Float64}
    _w4l::StrideOneVector{Float64}

    x_trial::Vector{Float64}
    c_trial::Vector{Float64}
    obj_val_trial::Float64

    x_slk::StrideOneVector{Float64}
    c_slk::SubVector{Float64}
    rhs::Vector{Float64}

    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    ind_llb::Vector{Int}
    ind_uub::Vector{Int}

    x_lr::SubVector{Float64}
    x_ur::SubVector{Float64}
    xl_r::SubVector{Float64}
    xu_r::SubVector{Float64}
    zl_r::SubVector{Float64}
    zu_r::SubVector{Float64}

    dx_lr::SubVector{Float64}
    dx_ur::SubVector{Float64}
    x_trial_lr::SubVector{Float64}
    x_trial_ur::SubVector{Float64}

    linear_solver::AbstractLinearSolver
    iterator::AbstractIterator

    obj_scale::Vector{Float64}
    con_scale::Vector{Float64}
    con_jac_scale::Vector{Float64}
    inf_pr::Float64
    inf_du::Float64
    inf_compl::Float64

    theta_min::Float64
    theta_max::Float64
    mu::Float64
    tau::Float64

    alpha::Float64
    alpha_z::Float64
    ftype::String

    del_w::Float64
    del_c::Float64
    del_w_last::Float64

    filter::Vector{Tuple{Float64,Float64}}

    RR::Union{Nothing,RobustRestorer}
    status::Status
    output::Dict
end

struct MadNLPExecutionStats{T} <: AbstractExecutionStats
    status::Status
    solution::StrideOneVector{T}
    objective::T
    constraints::StrideOneVector{T}
    dual_feas::T
    primal_feas::T
    multipliers::StrideOneVector{T}
    multipliers_L::StrideOneVector{T}
    multipliers_U::StrideOneVector{T}
    iter::Int
    counters::NLPModelsCounters
    elapsed_time::Real
end

struct InvalidNumberException <: Exception end
struct NotEnoughDegreesOfFreedomException <: Exception end

MadNLPExecutionStats(ips::InteriorPointSolver) =MadNLPExecutionStats(
    ips.status,view(ips.x,1:get_nvar(ips.nlp)),ips.obj_val,ips.c,
    ips.inf_du, ips.inf_pr,
    ips.l,view(ips.zl,1:get_nvar(ips.nlp)),view(ips.zu,1:get_nvar(ips.nlp)),
    ips.cnt.k, ips.nlp.counters,ips.cnt.total_time)
getStatus(result::MadNLPExecutionStats) = STATUS_OUTPUT_DICT[result.status]

function RobustRestorer(ips::AbstractInteriorPointSolver)

    nn = Vector{Float64}(undef,ips.m)
    zp = Vector{Float64}(undef,ips.m)
    zn = Vector{Float64}(undef,ips.m)
    dpp= Vector{Float64}(undef,ips.m)
    dnn= Vector{Float64}(undef,ips.m)
    dzp= Vector{Float64}(undef,ips.m)
    dzn= Vector{Float64}(undef,ips.m)
    pp_trial = Vector{Float64}(undef,ips.m)
    nn_trial = Vector{Float64}(undef,ips.m)

    return RobustRestorer(0.,ips._w2x,ips._w1x,0.,ips._w3x,0.,ips._w3l,ips._w4l,
                          zp,zn,dpp,dnn,dzp,dzn,ips._w2l,ips._w1l,
                          0.,0.,0.,0.,0.,0.,Tuple{Float64,Float64}[])
end

function initialize_robust_restorer!(ips::AbstractInteriorPointSolver)
    @trace(ips.logger,"Initializing restoration phase variables.")
    ips.RR == nothing && (ips.RR = RobustRestorer(ips))
    RR = ips.RR

    RR.x_ref .= ips.x
    RR.theta_ref = get_theta(ips.c)
    RR.D_R   .= min.(1,1 ./abs.(RR.x_ref))

    RR.mu_R = max(ips.mu,norm(ips.c,Inf))
    RR.tau_R= max(ips.opt.tau_min,1-RR.mu_R)
    RR.zeta = sqrt(RR.mu_R)

    RR.nn .= (RR.mu_R.-ips.opt.rho.*ips.c)./2 ./ips.opt.rho .+
        sqrt.(((RR.mu_R.-ips.opt.rho.*ips.c)./2 ./ips.opt.rho).^2 .+ RR.mu_R.*ips.c./2 ./ips.opt.rho)
    RR.pp .= ips.c .+ RR.nn
    RR.zp .= RR.mu_R./RR.pp
    RR.zn .= RR.mu_R./RR.nn

    RR.obj_val_R = get_obj_val_R(RR.pp,RR.nn,RR.D_R,ips.x,RR.x_ref,ips.opt.rho,RR.zeta)
    RR.f_R.=0
    empty!(RR.filter)
    push!(RR.filter,(ips.theta_max,-Inf))

    ips.l .= 0.
    ips.zl_r .= min.(ips.opt.rho,ips.zl_r)
    ips.zu_r .= min.(ips.opt.rho,ips.zu_r)
    ips.cnt.t = 0

    # misc
    ips.del_w = 0
end

function factorize_wrapper!(ips::InteriorPointSolver)
    @trace(ips.logger,"Factorization started.")
    build_kkt!(ips.kkt)
    ips.cnt.linear_solver_time += @elapsed factorize!(ips.linear_solver)
end

function solve_refine_wrapper!(ips::InteriorPointSolver, x,b)
    cnt = ips.cnt
    @trace(ips.logger,"Iterative solution started.")
    fixed_variable_treatment_vec!(b, ips.ind_fixed)

    cnt.linear_solver_time += @elapsed (result = solve_refine!(x, ips.iterator, b))
    if result == :Solved
        solve_status =  true
    else
        if improve!(ips.linear_solver)
            cnt.linear_solver_time += @elapsed begin
                factorize!(ips.linear_solver)
                solve_status = (solve_refine!(x, ips.iterator, b) == :Solved ? true : false)
            end
        else
            solve_status = false
        end
    end
    fixed_variable_treatment_vec!(x, ips.ind_fixed)
    return solve_status
end

function eval_f_wrapper(ips::InteriorPointSolver, x::Vector{Float64})
    nlp = ips.nlp
    cnt = ips.cnt
    @trace(ips.logger,"Evaluating objective.")
    cnt.eval_function_time += @elapsed obj_val = (get_minimize(nlp) ? 1. : -1.) * obj(nlp,view(x,1:get_nvar(nlp)))
    cnt.obj_cnt+=1
    cnt.obj_cnt==1 && (is_valid(obj_val) || throw(InvalidNumberException()))
    return obj_val*ips.obj_scale[]
end

function eval_grad_f_wrapper!(ips::InteriorPointSolver, f::Vector{Float64},x::Vector{Float64})
    nlp = ips.nlp
    cnt = ips.cnt
    @trace(ips.logger,"Evaluating objective gradient.")
    cnt.eval_function_time += @elapsed grad!(nlp,view(x,1:get_nvar(nlp)),view(f,1:get_nvar(nlp)))
    f.*=ips.obj_scale[] * (get_minimize(nlp) ? 1. : -1.)
    cnt.obj_grad_cnt+=1
    cnt.obj_grad_cnt==1 && (is_valid(f)  || throw(InvalidNumberException()))
    return f
end

function eval_cons_wrapper!(ips::InteriorPointSolver, c::Vector{Float64},x::Vector{Float64})
    nlp = ips.nlp
    cnt = ips.cnt
    @trace(ips.logger, "Evaluating constraints.")
    cnt.eval_function_time += @elapsed cons!(nlp,view(x,1:get_nvar(nlp)),view(c,1:get_ncon(nlp)))
    view(c,ips.ind_ineq).-=view(x,get_nvar(nlp)+1:ips.n)
    c.-=ips.rhs
    c.*=ips.con_scale
    cnt.con_cnt+=1
    cnt.con_cnt==2 && (is_valid(c) || throw(InvalidNumberException()))
    return c
end

function eval_jac_wrapper!(ipp::InteriorPointSolver, kkt::AbstractKKTSystem, x::Vector{Float64})
    nlp = ipp.nlp
    cnt = ipp.cnt
    ns = length(ipp.ind_ineq)
    @trace(ipp.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed jac_coord!(nlp,view(x,1:get_nvar(nlp)),jac)
    compress_jacobian!(kkt)
    cnt.con_jac_cnt+=1
    cnt.con_jac_cnt==1 && (is_valid(jac) || throw(InvalidNumberException()))
    @trace(ipp.logger,"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(ipp::InteriorPointSolver, kkt::AbstractKKTSystem, x::Vector{Float64},l::Vector{Float64};is_resto=false)
    nlp = ipp.nlp
    cnt = ipp.cnt
    @trace(ipp.logger,"Evaluating Lagrangian Hessian.")
    ipp._w1l .= l.*ipp.con_scale
    hess = get_hessian(kkt)
    cnt.eval_function_time += @elapsed hess_coord!(
        nlp, view(x,1:get_nvar(nlp)), ipp._w1l, hess;
        obj_weight = (get_minimize(nlp) ? 1. : -1.) * (is_resto ? 0.0 : ipp.obj_scale[]))
    compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1
    cnt.lag_hess_cnt==1 && (is_valid(hess) || throw(InvalidNumberException()))
    return hess
end

function eval_jac_wrapper!(ipp::InteriorPointSolver, kkt::DenseKKTSystem, x::Vector{Float64})
    nlp = ipp.nlp
    cnt = ipp.cnt
    ns = length(ipp.ind_ineq)
    @trace(ipp.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed jac_dense!(nlp,view(x,1:get_nvar(nlp)),jac)
    compress_jacobian!(kkt)
    cnt.con_jac_cnt+=1
    cnt.con_jac_cnt==1 && (is_valid(jac) || throw(InvalidNumberException()))
    @trace(ipp.logger,"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(ipp::InteriorPointSolver, kkt::DenseKKTSystem, x::Vector{Float64},l::Vector{Float64};is_resto=false)
    nlp = ipp.nlp
    cnt = ipp.cnt
    @trace(ipp.logger,"Evaluating Lagrangian Hessian.")
    ipp._w1l .= l.*ipp.con_scale
    hess = get_hessian(kkt)
    cnt.eval_function_time += @elapsed hess_dense!(
        nlp, view(x,1:get_nvar(nlp)), ipp._w1l, hess;
        obj_weight = (get_minimize(nlp) ? 1. : -1.) * (is_resto ? 0.0 : ipp.obj_scale[]))
    compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1
    cnt.lag_hess_cnt==1 && (is_valid(hess) || throw(InvalidNumberException()))
    return hess
end

function InteriorPointSolver(nlp::AbstractNLPModel;
    option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(), kwargs...
)
    opt = Options(linear_solver=default_linear_solver())
    set_options!(opt,option_dict,kwargs)
    check_option_sanity(opt)

    KKTSystem = if opt.kkt_system == SPARSE_KKT_SYSTEM
        MT = (opt.linear_solver.INPUT_MATRIX_TYPE == :csc) ? SparseMatrixCSC{Float64, Int32} : Matrix{Float64}
        SparseKKTSystem{Float64, MT}
    elseif opt.kkt_system == SPARSE_UNREDUCED_KKT_SYSTEM
        MT = (opt.linear_solver.INPUT_MATRIX_TYPE == :csc) ? SparseMatrixCSC{Float64, Int32} : Matrix{Float64}
        SparseUnreducedKKTSystem{Float64, MT}
    elseif opt.kkt_system == DENSE_KKT_SYSTEM
        MT = Matrix{Float64}
        VT = Vector{Float64}
        DenseKKTSystem{Float64, VT, MT}
    end
    return InteriorPointSolver{KKTSystem}(nlp, opt; option_linear_solver=option_dict)
end

# Inner constructor
function InteriorPointSolver{KKTSystem}(nlp::AbstractNLPModel, opt::Options;
    option_linear_solver::Dict{Symbol,Any}=Dict{Symbol,Any}(),
) where {KKTSystem<:AbstractKKTSystem}
    cnt = Counters(start_time=time())

    logger = Logger(print_level=opt.print_level,file_print_level=opt.file_print_level,
                    file = opt.output_file == "" ? nothing : open(opt.output_file,"w+"))
    @trace(logger,"Logger is initialized.")

    # generic options
    opt.disable_garbage_collector &&
        (GC.enable(false); @warn(logger,"Julia garbage collector is temporarily disabled"))
    set_blas_num_threads(opt.blas_num_threads; permanent=true)

    @trace(logger,"Initializing variables.")
    ind_cons = get_index_constraints(nlp; fixed_variable_treatment=opt.fixed_variable_treatment)
    ns = length(ind_cons.ind_ineq)
    n = get_nvar(nlp)+ns
    m = get_ncon(nlp)

    # Initialize KKT
    kkt = KKTSystem(nlp, ind_cons)

    xl = [get_lvar(nlp);view(get_lcon(nlp),ind_cons.ind_ineq)]
    xu = [get_uvar(nlp);view(get_ucon(nlp),ind_cons.ind_ineq)]
    x = [get_x0(nlp);zeros(ns)]
    l = copy(get_y0(nlp))
    zl= zeros(get_nvar(nlp)+ns)
    zu= zeros(get_nvar(nlp)+ns)

    f = zeros(n) # not sure why, but seems necessary to initialize to 0 when used with Plasmo interface
    c = zeros(m)

    n_jac = nnz_jacobian(kkt)

    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    x_trial=Vector{Float64}(undef,n)
    c_trial=Vector{Float64}(undef,m)

    x_slk= view(x,get_nvar(nlp)+1:n)
    c_slk= view(c,ind_cons.ind_ineq)
    rhs = (get_lcon(nlp).==get_ucon(nlp)).*get_lcon(nlp)

    x_lr = view(x, ind_cons.ind_lb)
    x_ur = view(x, ind_cons.ind_ub)
    xl_r = view(xl, ind_cons.ind_lb)
    xu_r = view(xu, ind_cons.ind_ub)
    zl_r = view(zl, ind_cons.ind_lb)
    zu_r = view(zu, ind_cons.ind_ub)
    x_trial_lr = view(x_trial, ind_cons.ind_lb)
    x_trial_ur = view(x_trial, ind_cons.ind_ub)

    aug_vec_length = is_reduced(kkt) ? n+m : n+m+nlb+nub

    _w1 = Vector{Float64}(undef,aug_vec_length)
    _w1x= view(_w1,1:n)
    _w1l= view(_w1,n+1:n+m)
    _w1zl = is_reduced(kkt) ? nothing : view(_w1,n+m+1:n+m+nlb)
    _w1zu = is_reduced(kkt) ? nothing : view(_w1,n+m+nlb+1:n+m+nlb+nub)


    _w2 = Vector{Float64}(undef,aug_vec_length)
    _w2x= view(_w2,1:n)
    _w2l= view(_w2,n+1:n+m)
    _w2zl = is_reduced(kkt) ? nothing : view(_w2,n+m+1:n+m+nlb)
    _w2zu = is_reduced(kkt) ? nothing : view(_w2,n+m+nlb+1:n+m+nlb+nub)

    _w3 = Vector{Float64}(undef,aug_vec_length)
    _w3x= view(_w3,1:n)
    _w3l= view(_w3,n+1:n+m)
    _w4 = zeros(aug_vec_length) # need to initialize to zero due to mul!
    _w4x= view(_w4,1:n)
    _w4l= view(_w4,n+1:n+m)

    jacl = zeros(n) # spblas may throw an error if not initialized to zero

    d = Vector{Float64}(undef,aug_vec_length)
    dx= view(d,1:n)
    dl= view(d,n+1:n+m)
    dzl= is_reduced(kkt) ? Vector{Float64}(undef,nlb) : view(d,n+m+1:n+m+nlb)
    dzu= is_reduced(kkt) ? Vector{Float64}(undef,nub) : view(d,n+m+nlb+1:n+m+nlb+nub)
    dx_lr = view(dx,ind_cons.ind_lb)
    dx_ur = view(dx,ind_cons.ind_ub)

    p = Vector{Float64}(undef,aug_vec_length)
    px= view(p,1:n)
    pl= view(p,n+1:n+m)
    pzl= is_reduced(kkt) ? Vector{Float64}(undef,nlb) : view(p,n+m+1:n+m+nlb)
    pzu= is_reduced(kkt) ? Vector{Float64}(undef,nub) : view(p,n+m+nlb+1:n+m+nlb+nub)

    obj_scale = [1.0]
    con_scale = ones(m)
    con_jac_scale = ones(n_jac)

    @trace(logger,"Initializing linear solver.")
    cnt.linear_solver_time =
        @elapsed linear_solver = opt.linear_solver.Solver(get_kkt(kkt) ; option_dict=option_linear_solver,logger=logger)

    @trace(logger,"Initializing iterative solver.")
    iterator = opt.iterator.Solver(
        similar(d),
        (b, x)->mul!(b, kkt, x), (x)->solve!(linear_solver, x) ; option_dict=option_linear_solver)

    @trace(logger,"Initializing fixed variable treatment scheme.")

    if opt.inertia_correction_method == INERTIA_AUTO
        opt.inertia_correction_method = is_inertia(linear_solver) ? INERTIA_BASED : INERTIA_FREE
    end

    !isempty(option_linear_solver) && print_ignored_options(logger, option_linear_solver)

    return InteriorPointSolver{KKTSystem}(nlp,kkt,opt,cnt,logger,
        n,m,nlb,nub,x,l,zl,zu,xl,xu,0.,f,c,
        jacl,
        d,dx,dl,dzl,dzu,p,px,pl,pzl,pzu,
        _w1,_w1x,_w1l,_w1zl,_w1zu,_w2,_w2x,_w2l,_w2zl,_w2zu,_w3,_w3x,_w3l,_w4,_w4x,_w4l,
        x_trial,c_trial,0.,x_slk,c_slk,rhs,
        ind_cons.ind_ineq,ind_cons.ind_fixed,ind_cons.ind_llb,ind_cons.ind_uub,
        x_lr,x_ur,xl_r,xu_r,zl_r,zu_r,dx_lr,dx_ur,x_trial_lr,x_trial_ur,
        linear_solver,iterator,
        obj_scale,con_scale,con_jac_scale,
        0.,0.,0.,0.,0.,0.,0.,0.,0.," ",0.,0.,0.,
        Vector{Float64}[],nothing,INITIAL,Dict(),
    )
end

function initialize!(ips::AbstractInteriorPointSolver)
    # initializing slack variables
    @trace(ips.logger,"Initializing slack variables.")
    cons!(ips.nlp,get_x0(ips.nlp),view(ips.c,1:get_ncon(ips.nlp)))
    ips.cnt.con_cnt += 1
    ips.x_slk.=ips.c_slk

    # Initialization
    @trace(ips.logger,"Initializing primal and bound duals.")
    ips.zl_r.=1.0
    ips.zu_r.=1.0
    ips.xl_r.-= max.(1,abs.(ips.xl_r)).*ips.opt.tol
    ips.xu_r.+= max.(1,abs.(ips.xu_r)).*ips.opt.tol
    initialize_variables!(ips.x,ips.xl,ips.xu,ips.opt.bound_push,ips.opt.bound_fac)

    # Automatic scaling (constraints)
    @trace(ips.logger,"Computing constraint scaling.")
    eval_jac_wrapper!(ips, ips.kkt, ips.x)
    compress_jacobian!(ips.kkt)
    if (ips.m > 0) && ips.opt.nlp_scaling
        jac = get_raw_jacobian(ips.kkt)
        set_con_scale!(ips.con_scale, jac, ips.opt.nlp_scaling_max_gradient)
        set_jacobian_scaling!(ips.kkt, ips.con_scale)
        ips.l./=ips.con_scale
    end
    compress_jacobian!(ips.kkt)

    # Automatic scaling (objective)
    eval_grad_f_wrapper!(ips, ips.f,ips.x)
    @trace(ips.logger,"Computing objective scaling.")
    if ips.opt.nlp_scaling
        ips.obj_scale[] = min(1,ips.opt.nlp_scaling_max_gradient/norm(ips.f,Inf))
        ips.f.*=ips.obj_scale[]
    end

    # Initialize dual variables
    @trace(ips.logger,"Initializing constraint duals.")
    if !ips.opt.dual_initialized
        set_initial_rhs!(ips, ips.kkt)
        initialize!(ips.kkt)
        factorize_wrapper!(ips)
        solve_refine_wrapper!(ips,ips.d,ips.p)
        norm(ips.dl,Inf)>ips.opt.constr_mult_init_max ? (ips.l.= 0.) : (ips.l.= ips.dl)
    end

    # Initializing
    ips.obj_val = eval_f_wrapper(ips, ips.x)
    eval_cons_wrapper!(ips, ips.c, ips.x)
    eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.l)

    theta = get_theta(ips.c)
    ips.theta_max=1e4*max(1,theta)
    ips.theta_min=1e-4*max(1,theta)
    ips.mu=ips.opt.mu_init
    ips.tau=max(ips.opt.tau_min,1-ips.opt.mu_init)
    ips.filter = [(ips.theta_max,-Inf)]

    return REGULAR
end


function reinitialize!(ips::AbstractInteriorPointSolver)
    view(ips.x,1:get_nvar(ips.nlp)) .= get_x0(ips.nlp)

    ips.obj_val = eval_f_wrapper(ips, ips.x)
    eval_grad_f_wrapper!(ips, ips.f, ips.x)
    eval_cons_wrapper!(ips, ips.c, ips.x)
    eval_jac_wrapper!(ips, ips.kkt, ips.x)
    eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.l)

    theta = get_theta(ips.c)
    ips.theta_max=1e4*max(1,theta)
    ips.theta_min=1e-4*max(1,theta)
    ips.mu=ips.opt.mu_init
    ips.tau=max(ips.opt.tau_min,1-ips.opt.mu_init)
    ips.filter = [(ips.theta_max,-Inf)]

    return REGULAR
end

# major loops ---------------------------------------------------------
function optimize!(ips::AbstractInteriorPointSolver)
    try
        if ips.status == INITIAL
            @notice(ips.logger,"This is $(introduce()), running with $(introduce(ips.linear_solver))\n")
            print_init(ips)
            ips.status = initialize!(ips)
        else # resolving the problem
            ips.status = reinitialize!(ips)
        end

        while ips.status >= REGULAR
            ips.status == REGULAR && (ips.status = regular!(ips))
            ips.status == RESTORE && (ips.status = restore!(ips))
            ips.status == ROBUST && (ips.status = robust!(ips))
        end
    catch e
        if e isa InvalidNumberException
            ips.status=INVALID_NUMBER_DETECTED
        elseif e isa NotEnoughDegreesOfFreedomException
            ips.status=NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa LinearSolverException
            ips.status=ERROR_IN_STEP_COMPUTATION;
            ips.opt.rethrow_error && rethrow(e)
        elseif e isa InterruptException
            ips.status=USER_REQUESTED_STOP
            ips.opt.rethrow_error && rethrow(e)
        else
            ips.status=INTERNAL_ERROR
            ips.opt.rethrow_error && rethrow(e)
        end
    finally
        ips.cnt.total_time = time() - ips.cnt.start_time
        !(ips.status < SOLVE_SUCCEEDED) && (print_summary_1(ips);print_summary_2(ips))
        # Unscale once the summary has been printed out
        unscale!(ips)
        @notice(ips.logger,"EXIT: $(STATUS_OUTPUT_DICT[ips.status])")
        ips.opt.disable_garbage_collector &&
            (GC.enable(true); @warn(ips.logger,"Julia garbage collector is turned back on"))
        finalize(ips.logger)
    end
    return MadNLPExecutionStats(ips)
end

function unscale!(ips::AbstractInteriorPointSolver)
    ips.obj_val/=ips.obj_scale[]
    ips.c ./= ips.con_scale
    ips.c .-= ips.rhs
    ips.c_slk .+= ips.x_slk
end

function regular!(ips::AbstractInteriorPointSolver)
    while true
        if (ips.cnt.k!=0 && !ips.opt.jacobian_constant)
            eval_jac_wrapper!(ips, ips.kkt, ips.x)
        end
        jtprod!(ips.jacl, ips.kkt, ips.l)
        fixed_variable_treatment_vec!(ips.jacl,ips.ind_fixed)
        fixed_variable_treatment_z!(ips.zl,ips.zu,ips.f,ips.jacl,ips.ind_fixed)

        sd = get_sd(ips.l,ips.zl_r,ips.zu_r,ips.opt.s_max)
        sc = get_sc(ips.zl_r,ips.zu_r,ips.opt.s_max)

        ips.inf_pr = get_inf_pr(ips.c)
        ips.inf_du = get_inf_du(ips.f,ips.zl,ips.zu,ips.jacl,sd)
        ips.inf_compl = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,0.,sc)
        inf_compl_mu = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,ips.mu,sc)

        print_iter(ips)

        # evaluate termination criteria
        @trace(ips.logger,"Evaluating termination criteria.")
        max(ips.inf_pr,ips.inf_du,ips.inf_compl) <= ips.opt.tol && return SOLVE_SUCCEEDED
        max(ips.inf_pr,ips.inf_du,ips.inf_compl) <= ips.opt.acceptable_tol ?
            (ips.cnt.acceptable_cnt < ips.opt.acceptable_iter ?
            ips.cnt.acceptable_cnt+=1 : return SOLVED_TO_ACCEPTABLE_LEVEL) : (ips.cnt.acceptable_cnt = 0)
        max(ips.inf_pr,ips.inf_du,ips.inf_compl) >= ips.opt.diverging_iterates_tol && return DIVERGING_ITERATES
        ips.cnt.k>=ips.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-ips.cnt.start_time>=ips.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        @trace(ips.logger,"Updating the barrier parameter.")
        while ips.mu != max(ips.opt.mu_min,ips.opt.tol/10) &&
            max(ips.inf_pr,ips.inf_du,inf_compl_mu) <= ips.opt.barrier_tol_factor*ips.mu
            mu_new = get_mu(ips.mu,ips.opt.mu_min,
                            ips.opt.mu_linear_decrease_factor,ips.opt.mu_superlinear_decrease_power,ips.opt.tol)
            inf_compl_mu = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,ips.mu,sc)
            ips.tau= get_tau(ips.mu,ips.opt.tau_min)
            ips.mu = mu_new
            empty!(ips.filter)
            push!(ips.filter,(ips.theta_max,-Inf))
        end

        # compute the newton step
        @trace(ips.logger,"Computing the newton step.")
        if (ips.cnt.k!=0 && !ips.opt.hessian_constant)
            eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.l)
        end

        set_aug_diagonal!(ips.kkt,ips)
        set_aug_rhs!(ips, ips.kkt, ips.c)
        if ips.opt.inertia_correction_method == INERTIA_FREE
            set_aug_rhs_ifr!(ips, ips.kkt)
        end
        dual_inf_perturbation!(ips.px,ips.ind_llb,ips.ind_uub,ips.mu,ips.opt.kappa_d)

        # start inertia conrrection
        @trace(ips.logger,"Solving primal-dual system.")
        if ips.opt.inertia_correction_method == INERTIA_FREE
            inertia_free_reg(ips) || return ROBUST
        elseif ips.opt.inertia_correction_method == INERTIA_BASED
            inertia_based_reg(ips) || return ROBUST
        end

        finish_aug_solve!(ips, ips.kkt, ips.mu)

        # filter start
        @trace(ips.logger,"Backtracking line search initiated.")
        theta = get_theta(ips.c)
        varphi= get_varphi(ips.obj_val,ips.x_lr,ips.xl_r,ips.xu_r,ips.x_ur,ips.mu)
        varphi_d = get_varphi_d(ips.f,ips.x,ips.xl,ips.xu,ips.dx,ips.mu)


        alpha_max = get_alpha_max(ips.x,ips.xl,ips.xu,ips.dx,ips.tau)
        ips.alpha_z = get_alpha_z(ips.zl_r,ips.zu_r,ips.dzl,ips.dzu,ips.tau)
        alpha_min = get_alpha_min(theta,varphi_d,ips.theta_min,ips.opt.gamma_theta,ips.opt.gamma_phi,
                                  ips.opt.alpha_min_frac,ips.opt.delta,ips.opt.s_theta,ips.opt.s_phi)
        ips.cnt.l = 1
        ips.alpha = alpha_max
        varphi_trial= 0.
            theta_trial = 0.
            small_search_norm = get_rel_search_norm(ips.x,ips.dx) < 10*eps(Float64)
        switching_condition = is_switching(varphi_d,ips.alpha,ips.opt.s_phi,ips.opt.delta,2.,ips.opt.s_theta)
        armijo_condition = false
        while true
            copyto!(ips.x_trial,ips.x)
            axpy!(ips.alpha,ips.dx,ips.x_trial)

            ips.obj_val_trial = eval_f_wrapper(ips, ips.x_trial)
            eval_cons_wrapper!(ips, ips.c_trial, ips.x_trial)

            theta_trial = get_theta(ips.c_trial)
            varphi_trial= get_varphi(ips.obj_val_trial,ips.x_trial_lr,ips.xl_r,ips.xu_r,ips.x_trial_ur,ips.mu)
            armijo_condition = is_armijo(varphi_trial,varphi,ips.opt.eta_phi,ips.alpha,varphi_d)

            small_search_norm && break

            ips.ftype = get_ftype(
                ips.filter,theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
                ips.theta_min,ips.opt.obj_max_inc,ips.opt.gamma_theta,ips.opt.gamma_phi,
                has_constraints(ips))
            ips.ftype in ["f","h"] && (@trace(ips.logger,"Step accepted with type $(ips.ftype)"); break)

            ips.cnt.l==1 && theta_trial>=theta && second_order_correction(
                ips,alpha_max,theta,varphi,theta_trial,varphi_d,switching_condition) && break

            ips.alpha /= 2
            ips.cnt.l += 1
            if ips.alpha < alpha_min
                @debug(ips.logger,
                       "Cannot find an acceptable step at iteration $(ips.cnt.k). Switching to restoration phase.")
                ips.cnt.k+=1
                return RESTORE
            else
                @trace(ips.logger,"Step rejected; proceed with the next trial step.")
                ips.alpha * norm(ips.dx) < eps(Float64)*10 &&
                    return ips.cnt.acceptable_cnt >0 ?
                    SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
            end
        end

        @trace(ips.logger,"Updating primal-dual variables.")
        ips.x.=ips.x_trial
        ips.c.=ips.c_trial
        ips.obj_val=ips.obj_val_trial
        adjusted = adjust_boundary!(ips.x_lr,ips.xl_r,ips.x_ur,ips.xu_r,ips.mu)
        adjusted > 0 &&
            @warn(ips.logger,"In iteration $(ips.cnt.k), $adjusted Slack too small, adjusting variable bound")

        axpy!(ips.alpha,ips.dl,ips.l)
        axpy!(ips.alpha_z,ips.dzl,ips.zl_r)
        axpy!(ips.alpha_z,ips.dzu,ips.zu_r)
        reset_bound_dual!(ips.zl,ips.x,ips.xl,ips.mu,ips.opt.kappa_sigma)
        reset_bound_dual!(ips.zu,ips.xu,ips.x,ips.mu,ips.opt.kappa_sigma)
        eval_grad_f_wrapper!(ips, ips.f,ips.x)

        if !switching_condition || !armijo_condition
            @trace(ips.logger,"Augmenting filter.")
            augment_filter!(ips.filter,theta_trial,varphi_trial,ips.opt.gamma_theta)
        end

        ips.cnt.k+=1
        @trace(ips.logger,"Proceeding to the next interior point iteration.")
    end
end

function restore!(ips::AbstractInteriorPointSolver)
    ips.del_w=0
    ips._w1x .= ips.x # backup the previous primal iterate
    ips._w1l .= ips.l # backup the previous primal iterate
    ips._w2l .= ips.c # backup the previous primal iterate

    F = get_F(ips.c,ips.f,ips.zl,ips.zu,ips.jacl,ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,ips.mu)
    ips.cnt.t = 0
    ips.alpha_z = 0.
    ips.ftype = "R"

    while true
        ips.alpha = min(get_alpha_max(ips.x,ips.xl,ips.xu,ips.dx,ips.tau),
                        get_alpha_z(ips.zl_r,ips.zu_r,ips.dzl,ips.dzu,ips.tau))

        ips.x .+= ips.alpha.*ips.dx
        ips.l .+= ips.alpha.*ips.dl
        ips.zl_r.+=ips.alpha.*ips.dzl
        ips.zu_r.+=ips.alpha.*ips.dzu

        eval_cons_wrapper!(ips,ips.c,ips.x)
        eval_grad_f_wrapper!(ips,ips.f,ips.x)
        ips.obj_val = eval_f_wrapper(ips,ips.x)

        !ips.opt.jacobian_constant && eval_jac_wrapper!(ips,ips.kkt,ips.x)
        jtprod!(ips.jacl,ips.kkt,ips.l)

        F_trial = get_F(
            ips.c,ips.f,ips.zl,ips.zu,ips.jacl,ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,ips.mu)
        if F_trial > ips.opt.soft_resto_pderror_reduction_factor*F
            ips.x.=ips._w1x
            ips.l.=ips._w1l
            ips.c.=ips._w2l # backup the previous primal iterate
            return ROBUST
        end

        adjusted = adjust_boundary!(ips.x_lr,ips.xl_r,ips.x_ur,ips.xu_r,ips.mu)
        adjusted > 0 &&
            @warn(ips.logger,"In iteration $(ips.cnt.k), $adjusted Slack too small, adjusting variable bound")


        F = F_trial

        theta = get_theta(ips.c)
        varphi= get_varphi(ips.obj_val,ips.x_lr,ips.xl_r,ips.xu_r,ips.x_ur,ips.mu)

        ips.cnt.k+=1

        is_filter_acceptable(ips.filter,theta,varphi) ? (return REGULAR) : (ips.cnt.t+=1)
        ips.cnt.k>=ips.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-ips.cnt.start_time>=ips.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED


        sd = get_sd(ips.l,ips.zl_r,ips.zu_r,ips.opt.s_max)
        sc = get_sc(ips.zl_r,ips.zu_r,ips.opt.s_max)
        ips.inf_pr = get_inf_pr(ips.c)
        ips.inf_du = get_inf_du(ips.f,ips.zl,ips.zu,ips.jacl,sd)

        ips.inf_compl = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,0.,sc)
        inf_compl_mu = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,ips.mu,sc)
        print_iter(ips)

        !ips.opt.hessian_constant && eval_lag_hess_wrapper!(ips,ips.kkt,ips.x,ips.l)
        set_aug_diagonal!(ips.kkt,ips)
        set_aug_rhs!(ips, ips.kkt, ips.c)

        dual_inf_perturbation!(ips.px,ips.ind_llb,ips.ind_uub,ips.mu,ips.opt.kappa_d)
        factorize_wrapper!(ips)
        solve_refine_wrapper!(ips,ips.d,ips.p)
        finish_aug_solve!(ips, ips.kkt, ips.mu)

        ips.ftype = "f"
    end
end

function robust!(ips::InteriorPointSolver)
    initialize_robust_restorer!(ips)
    RR = ips.RR
    while true
        if !ips.opt.jacobian_constant
            eval_jac_wrapper!(ips, ips.kkt, ips.x)
        end
        jtprod!(ips.jacl, ips.kkt, ips.l)
        fixed_variable_treatment_vec!(ips.jacl,ips.ind_fixed)
        fixed_variable_treatment_z!(ips.zl,ips.zu,ips.f,ips.jacl,ips.ind_fixed)

        # evaluate termination criteria
        @trace(ips.logger,"Evaluating restoration phase termination criteria.")
        sd = get_sd(ips.l,ips.zl_r,ips.zu_r,ips.opt.s_max)
        sc = get_sc(ips.zl_r,ips.zu_r,ips.opt.s_max)
        ips.inf_pr = get_inf_pr(ips.c)
        ips.inf_du = get_inf_du(ips.f,ips.zl,ips.zu,ips.jacl,sd)
        ips.inf_compl = get_inf_compl(ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,0.,sc)

        # Robust restoration phase error
        RR.inf_pr_R = get_inf_pr_R(ips.c,RR.pp,RR.nn)
        RR.inf_du_R = get_inf_du_R(RR.f_R,ips.l,ips.zl,ips.zu,ips.jacl,RR.zp,RR.zn,ips.opt.rho,sd)
        RR.inf_compl_R = get_inf_compl_R(
            ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,0.,sc)
        inf_compl_mu_R = get_inf_compl_R(
            ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,sc)

        print_iter(ips;is_resto=true)

        max(RR.inf_pr_R,RR.inf_du_R,RR.inf_compl_R) <= ips.opt.tol && return INFEASIBLE_PROBLEM_DETECTED
        ips.cnt.k>=ips.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-ips.cnt.start_time>=ips.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED


        # update the barrier parameter
        @trace(ips.logger,"Updating restoration phase barrier parameter.")
        while RR.mu_R != ips.opt.mu_min*100 &&
            max(RR.inf_pr_R,RR.inf_du_R,inf_compl_mu_R) <= ips.opt.barrier_tol_factor*RR.mu_R
            RR.mu_R = get_mu(RR.mu_R,ips.opt.mu_min,
                            ips.opt.mu_linear_decrease_factor,ips.opt.mu_superlinear_decrease_power,ips.opt.tol)
            inf_compl_mu_R = get_inf_compl_R(
                ips.x_lr,ips.xl_r,ips.zl_r,ips.xu_r,ips.x_ur,ips.zu_r,RR.pp,RR.zp,RR.nn,RR.zn,RR.mu_R,sc)
            RR.tau_R= max(ips.opt.tau_min,1-RR.mu_R)
            RR.zeta = sqrt(RR.mu_R)

            empty!(RR.filter)
            push!(RR.filter,(ips.theta_max,-Inf))
        end

        # compute the newton step
        if !ips.opt.hessian_constant
            eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.l; is_resto=true)
        end
        set_aug_RR!(ips.kkt, ips, RR)
        set_aug_rhs_RR!(ips, ips.kkt, RR, ips.opt.rho)

        # without inertia correction,
        @trace(ips.logger,"Solving restoration phase primal-dual system.")
        factorize_wrapper!(ips)
        solve_refine_wrapper!(ips,ips.d,ips.p)

        finish_aug_solve!(ips, ips.kkt, RR.mu_R)
        finish_aug_solve_RR!(RR.dpp,RR.dnn,RR.dzp,RR.dzn,ips.l,ips.dl,RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,ips.opt.rho)


        theta_R = get_theta_R(ips.c,RR.pp,RR.nn)
        varphi_R = get_varphi_R(RR.obj_val_R,ips.x_lr,ips.xl_r,ips.xu_r,ips.x_ur,RR.pp,RR.nn,RR.mu_R)
        varphi_d_R = get_varphi_d_R(RR.f_R,ips.x,ips.xl,ips.xu,ips.dx,RR.pp,RR.nn,RR.dpp,RR.dnn,RR.mu_R,ips.opt.rho)

        # set alpha_min
        alpha_max = get_alpha_max_R(ips.x,ips.xl,ips.xu,ips.dx,RR.pp,RR.dpp,RR.nn,RR.dnn,RR.tau_R)
        ips.alpha_z = get_alpha_z_R(ips.zl_r,ips.zu_r,ips.dzl,ips.dzu,RR.zp,RR.dzp,RR.zn,RR.dzn,RR.tau_R)
        alpha_min = get_alpha_min(theta_R,varphi_d_R,ips.theta_min,ips.opt.gamma_theta,ips.opt.gamma_phi,
                                  ips.opt.alpha_min_frac,ips.opt.delta,ips.opt.s_theta,ips.opt.s_phi)

        # filter start
        @trace(ips.logger,"Backtracking line search initiated.")
        ips.alpha = alpha_max
        ips.cnt.l = 1
        theta_R_trial = 0.
        varphi_R_trial = 0.
        small_search_norm = get_rel_search_norm(ips.x,ips.dx) < 10*eps(Float64)
        switching_condition = is_switching(varphi_d_R,ips.alpha,ips.opt.s_phi,ips.opt.delta,theta_R,ips.opt.s_theta)
        armijo_condition = false

        while true
            copyto!(ips.x_trial,ips.x)
            copyto!(RR.pp_trial,RR.pp)
            copyto!(RR.nn_trial,RR.nn)
            axpy!(ips.alpha,ips.dx,ips.x_trial)
            axpy!(ips.alpha,RR.dpp,RR.pp_trial)
            axpy!(ips.alpha,RR.dnn,RR.nn_trial)

            RR.obj_val_R_trial = get_obj_val_R(
                RR.pp_trial,RR.nn_trial,RR.D_R,ips.x_trial,RR.x_ref,ips.opt.rho,RR.zeta)
            eval_cons_wrapper!(ips, ips.c_trial, ips.x_trial)
            theta_R_trial  = get_theta_R(ips.c_trial,RR.pp_trial,RR.nn_trial)
            varphi_R_trial = get_varphi_R(
                RR.obj_val_R_trial,ips.x_trial_lr,ips.xl_r,ips.xu_r,ips.x_trial_ur,RR.pp_trial,RR.nn_trial,RR.mu_R)

            armijo_condition = is_armijo(varphi_R_trial,varphi_R,0.,ips.alpha,varphi_d_R) #####

            small_search_norm && break
            ips.ftype = get_ftype(
                RR.filter,theta_R,theta_R_trial,varphi_R,varphi_R_trial,
                switching_condition,armijo_condition,
                ips.theta_min,ips.opt.obj_max_inc,ips.opt.gamma_theta,ips.opt.gamma_phi,
                has_constraints(ips))
            ips.ftype in ["f","h"] && (@trace(ips.logger,"Step accepted with type $(ips.ftype)"); break)

            ips.alpha /= 2
            ips.cnt.l += 1
            if ips.alpha < alpha_min
                @debug(ips.logger,"Restoration phase cannot find an acceptable step at iteration $(ips.cnt.k).")
                return RESTORATION_FAILED
            else
                @trace(ips.logger,"Step rejected; proceed with the next trial step.")
                ips.alpha < eps(Float64)*10 && return ips.cnt.acceptable_cnt >0 ?
                    SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
            end
        end

        @trace(ips.logger,"Updating primal-dual variables.")
        ips.x.=ips.x_trial
        ips.c.=ips.c_trial
        RR.pp.=RR.pp_trial
        RR.nn.=RR.nn_trial

        RR.obj_val_R=RR.obj_val_R_trial
        RR.f_R .= RR.zeta.*RR.D_R.^2 .*(ips.x.-RR.x_ref)

        axpy!(ips.alpha, ips.dl,ips.l )
        axpy!(ips.alpha_z, ips.dzl,ips.zl_r)
        axpy!(ips.alpha_z, ips.dzu,ips.zu_r)
        axpy!(ips.alpha_z, RR.dzp,RR.zp)
        axpy!(ips.alpha_z, RR.dzn,RR.zn)

        reset_bound_dual!(ips.zl,ips.x,ips.xl,RR.mu_R,ips.opt.kappa_sigma)
        reset_bound_dual!(ips.zu,ips.xu,ips.x,RR.mu_R,ips.opt.kappa_sigma)
        reset_bound_dual!(RR.zp,RR.pp,RR.mu_R,ips.opt.kappa_sigma)
        reset_bound_dual!(RR.zn,RR.nn,RR.mu_R,ips.opt.kappa_sigma)

        adjusted = adjust_boundary!(ips.x_lr,ips.xl_r,ips.x_ur,ips.xu_r,ips.mu)
        adjusted > 0 &&
            @warn(ips.logger,"In iteration $(ips.cnt.k), $adjusted Slack too small, adjusting variable bound")

        if !switching_condition || !armijo_condition
            @trace(ips.logger,"Augmenting restoration phase filter.")
            augment_filter!(RR.filter,theta_R_trial,varphi_R_trial,ips.opt.gamma_theta)
        end

        # check if going back to regular phase
        @trace(ips.logger,"Checking if going back to regular phase.")
        ips.obj_val = eval_f_wrapper(ips, ips.x)
        eval_grad_f_wrapper!(ips, ips.f, ips.x)
        theta = get_theta(ips.c)
        varphi= get_varphi(ips.obj_val,ips.x_lr,ips.xl_r,ips.xu_r,ips.x_ur,ips.mu)

        if is_filter_acceptable(ips.filter,theta,varphi) &&
            theta <= ips.opt.required_infeasibility_reduction * RR.theta_ref

            @trace(ips.logger,"Going back to the regular phase.")
            ips.zl_r.=1
            ips.zu_r.=1

            set_initial_rhs!(ips, ips.kkt)
            initialize!(ips.kkt)

            factorize_wrapper!(ips)
            solve_refine_wrapper!(ips,ips.d,ips.p)
            norm(ips.dl,Inf)>ips.opt.constr_mult_init_max ? (ips.l.= 0) : (ips.l.= ips.dl)
            ips.cnt.k+=1

            return REGULAR
        end

        ips.cnt.k>=ips.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-ips.cnt.start_time>=ips.opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        @trace(ips.logger,"Proceeding to the next restoration phase iteration.")
        ips.cnt.k+=1
        ips.cnt.t+=1
    end
end

function inertia_based_reg(ips::InteriorPointSolver)
    @trace(ips.logger,"Inertia-based regularization started.")

    factorize_wrapper!(ips)
    num_pos,num_zero,num_neg = inertia(ips.linear_solver)
    solve_status = num_zero!= 0 ? false : solve_refine_wrapper!(ips,ips.d,ips.p)

    n_trial = 0
    ips.del_w = del_w_prev = 0.0
    while !is_inertia_correct(ips.kkt, num_pos, num_zero, num_neg) || !solve_status
        @debug(ips.logger,"Primal-dual perturbed.")
        if ips.del_w == 0.0
            ips.del_w = ips.del_w_last==0. ? ips.opt.first_hessian_perturbation :
                max(ips.opt.min_hessian_perturbation,ips.opt.perturb_dec_fact*ips.del_w_last)
        else
            ips.del_w*= ips.del_w_last==0. ? ips.opt.perturb_inc_fact_first : ips.opt.perturb_inc_fact
            if ips.del_w>ips.opt.max_hessian_perturbation ips.cnt.k+=1
                @debug(ips.logger,"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        ips.del_c = (num_zero == 0 || !solve_status) ?
            ips.opt.jacobian_regularization_value * ips.mu^(ips.opt.jacobian_regularization_exponent) : 0.
        regularize_diagonal!(ips.kkt, ips.del_w - del_w_prev, ips.del_c)
        del_w_prev = ips.del_w

        factorize_wrapper!(ips)
        num_pos,num_zero,num_neg = inertia(ips.linear_solver)
        solve_status = num_zero!= 0 ? false : solve_refine_wrapper!(ips,ips.d,ips.p)
        n_trial += 1
    end
    ips.del_w != 0 && (ips.del_w_last = ips.del_w)

    return true
end


function inertia_free_reg(ips::InteriorPointSolver)

    @trace(ips.logger,"Inertia-free regularization started.")
    p0 = ips._w1
    d0= ips._w2
    t = ips._w3x
    n = ips._w2x
    wx= ips._w4x
    ips._w3l.=0

    g = ips.x_trial # just to avoid new allocation
    g .= ips.f.-ips.mu./(ips.x.-ips.xl).+ips.mu./(ips.xu.-ips.x).+ips.jacl

    fixed_variable_treatment_vec!(ips._w1x,ips.ind_fixed)
    fixed_variable_treatment_vec!(ips.px,ips.ind_fixed)
    fixed_variable_treatment_vec!(g,ips.ind_fixed)
    # end

    factorize_wrapper!(ips)
    solve_status = (solve_refine_wrapper!(ips,d0,p0) && solve_refine_wrapper!(ips,ips.d,ips.p))
    t .= ips.dx.-n
    mul!(ips._w4, ips.kkt, ips._w3) # prepartation for curv_test
    n_trial = 0
    ips.del_w = del_w_prev = 0.

    while !curv_test(t,n,g,wx,ips.opt.inertia_free_tol)  || !solve_status
        @debug(ips.logger,"Primal-dual perturbed.")
        if n_trial == 0
            ips.del_w = ips.del_w_last==.0 ? ips.opt.first_hessian_perturbation :
                max(ips.opt.min_hessian_perturbation,ips.opt.perturb_dec_fact*ips.del_w_last)
        else
            ips.del_w*= ips.del_w_last==.0 ? ips.opt.perturb_inc_fact_first : ips.opt.perturb_inc_fact
            if ips.del_w>ips.opt.max_hessian_perturbation ips.cnt.k+=1
                @debug(ips.logger,"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        ips.del_c = !solve_status ?
            ips.opt.jacobian_regularization_value * ips.mu^(ips.opt.jacobian_regularization_exponent) : 0.
        regularize_diagonal!(ips.kkt, ips.del_w - del_w_prev, ips.del_c)
        del_w_prev = ips.del_w

        factorize_wrapper!(ips)
        solve_status = (solve_refine_wrapper!(ips,d0,p0) && solve_refine_wrapper!(ips,ips.d,ips.p))
        t .= ips.dx.-n
        mul!(ips._w4, ips.kkt, ips._w3) # prepartation for curv_test
        n_trial += 1
    end

    ips.del_w != 0 && (ips.del_w_last = ips.del_w)
    return true
end

curv_test(t,n,g,wx,inertia_free_tol) = dot(wx,t) + max(dot(wx,n)-dot(g,n),0) - inertia_free_tol*dot(t,t) >=0

function second_order_correction(ips::AbstractInteriorPointSolver,alpha_max::Float64,theta::Float64,varphi::Float64,
                                 theta_trial::Float64,varphi_d::Float64,switching_condition::Bool)
    @trace(ips.logger,"Second-order correction started.")

    ips._w1l .= alpha_max .* ips.c .+ ips.c_trial
    theta_soc_old = theta_trial
    for p=1:ips.opt.max_soc
        # compute second order correction
        set_aug_rhs!(ips, ips.kkt, ips._w1l)
        dual_inf_perturbation!(ips.px,ips.ind_llb,ips.ind_uub,ips.mu,ips.opt.kappa_d)
        solve_refine_wrapper!(ips,ips._w1,ips.p)
        alpha_soc = get_alpha_max(ips.x,ips.xl,ips.xu,ips._w1x,ips.tau)

        ips.x_trial .= ips.x.+alpha_soc.*ips._w1x
        eval_cons_wrapper!(ips, ips.c_trial,ips.x_trial)
        ips.obj_val_trial = eval_f_wrapper(ips, ips.x_trial)

        theta_soc = get_theta(ips.c_trial)
        varphi_soc= get_varphi(ips.obj_val_trial,ips.x_trial_lr,ips.xl_r,ips.xu_r,ips.x_trial_ur,ips.mu)

        !is_filter_acceptable(ips.filter,theta_soc,varphi_soc) && break

        if theta <=ips.theta_min && switching_condition
            # Case I
            if is_armijo(varphi_soc,varphi,ips.opt.eta_phi,ips.alpha,varphi_d)
                @trace(ips.logger,"Step in second order correction accepted by armijo condition.")
                ips.ftype = "F"
                ips.alpha=alpha_soc
                return true
            end
        else
            # Case II
            if is_sufficient_progress(theta_soc,theta,ips.opt.gamma_theta,varphi_soc,varphi,ips.opt.gamma_phi,has_constraints(ips))
                @trace(ips.logger,"Step in second order correction accepted by sufficient progress.")
                ips.ftype = "H"
                ips.alpha=alpha_soc
                return true
            end
        end

        theta_soc>ips.opt.kappa_soc*theta_soc_old && break
        theta_soc_old = theta_soc
    end
    @trace(ips.logger,"Second-order correction terminated.")

    return false
end


# KKT system updates -------------------------------------------------------
# Set diagonal
function set_aug_diagonal!(kkt::AbstractKKTSystem, ips::InteriorPointSolver)
    kkt.pr_diag .= ips.zl./(ips.x.-ips.xl) .+ ips.zu./(ips.xu.-ips.x)
    fill!(kkt.du_diag, 0.0)
end
function set_aug_diagonal!(kkt::SparseUnreducedKKTSystem, ips::InteriorPointSolver)
    kkt.pr_diag .= 0.0
    kkt.du_diag .= 0.0
    kkt.l_lower .= .-sqrt.(ips.zl_r)
    kkt.u_lower .= .-sqrt.(ips.zu_r)
    kkt.l_diag  .= ips.xl_r .- ips.x_lr
    kkt.u_diag  .= ips.x_ur .- ips.xu_r
end

# Robust restoration
function set_aug_RR!(kkt::AbstractKKTSystem, ips::InteriorPointSolver, RR::RobustRestorer)
    kkt.pr_diag .= ips.zl./(ips.x.-ips.xl) .+ ips.zu./(ips.xu.-ips.x) .+ RR.zeta.*RR.D_R.^2
    kkt.du_diag .= .-RR.pp./RR.zp .- RR.nn./RR.zn
end
function set_aug_RR!(kkt::SparseUnreducedKKTSystem, ips::InteriorPointSolver, RR::RobustRestorer)
    kkt.pr_diag.= RR.zeta.*RR.D_R.^2
    kkt.du_diag.= .-RR.pp./RR.zp.-RR.nn./RR.zn
    kkt.l_lower.=.-sqrt.(ips.zl_r)
    kkt.u_lower.=.-sqrt.(ips.zu_r)
    kkt.l_diag .= ips.xl_r .- ips.x_lr
    kkt.u_diag .= ips.x_ur .- ips.xu_r
end

# Set RHS
function set_aug_rhs!(ips::InteriorPointSolver, kkt::AbstractKKTSystem, c)
    ips.px.=.-ips.f.+ips.mu./(ips.x.-ips.xl).-ips.mu./(ips.xu.-ips.x).-ips.jacl
    ips.pl.=.-c
end

function set_aug_rhs!(ips::InteriorPointSolver, kkt::SparseUnreducedKKTSystem, c)
    ips.px.=.-ips.f.+ips.zl.-ips.zu.-ips.jacl
    ips.pl.=.-c
    ips.pzl.=(ips.xl_r-ips.x_lr).*kkt.l_lower .+ ips.mu./kkt.l_lower
    ips.pzu.=(ips.xu_r-ips.x_ur).*kkt.u_lower .- ips.mu./kkt.u_lower
end

function set_aug_rhs_ifr!(ips::InteriorPointSolver, kkt::SparseUnreducedKKTSystem,c)
    ips._w1x .= 0.
    ips._w1l .= .-c
    ips._w1zl.= 0.
    ips._w1zu.= 0.
end

# Set RHS RR
function set_aug_rhs_RR!(
    ips::InteriorPointSolver, kkt::AbstractKKTSystem, RR::RobustRestorer, rho,
)
    ips.px.=.-RR.f_R.-ips.jacl.+RR.mu_R./(ips.x.-ips.xl).-RR.mu_R./(ips.xu.-ips.x)
    ips.pl.=.-ips.c.+RR.pp.-RR.nn.+(RR.mu_R.-(rho.-ips.l).*RR.pp)./RR.zp.-(RR.mu_R.-(rho.+ips.l).*RR.nn)./RR.zn
end

# Finish
function finish_aug_solve!(ips::InteriorPointSolver, kkt::AbstractKKTSystem, mu)
    ips.dzl.= (mu.-ips.zl_r.*ips.dx_lr)./(ips.x_lr.-ips.xl_r).-ips.zl_r
    ips.dzu.= (mu.+ips.zu_r.*ips.dx_ur)./(ips.xu_r.-ips.x_ur).-ips.zu_r
end

function finish_aug_solve!(ips::InteriorPointSolver, kkt::SparseUnreducedKKTSystem, mu)
    ips.dzl.*=.-kkt.l_lower
    ips.dzu.*=kkt.u_lower
    ips.dzl.= (mu.-ips.zl_r.*ips.dx_lr)./(ips.x_lr.-ips.xl_r).-ips.zl_r
    ips.dzu.= (mu.+ips.zu_r.*ips.dx_ur)./(ips.xu_r.-ips.x_ur).-ips.zu_r
end

# Initial
function set_initial_rhs!(ips::InteriorPointSolver, kkt::AbstractKKTSystem)
    ips.px .= .-ips.f.+ips.zl.-ips.zu
    ips.pl .= 0.0
end
function set_initial_rhs!(ips::InteriorPointSolver, kkt::SparseUnreducedKKTSystem)
    ips.px .= .-ips.f.+ips.zl.-ips.zu
    ips.pl .= 0.0
    ips.pzl.= 0.0
    ips.pzu.= 0.0
end

# Set ifr
function set_aug_rhs_ifr!(ips::InteriorPointSolver, kkt::AbstractKKTSystem)
    ips._w1x .= 0.0
    ips._w1l .= .-ips.c
end

# Finish RR
function finish_aug_solve_RR!(dpp,dnn,dzp,dzn,l,dl,pp,nn,zp,zn,mu_R,rho)
    dpp .= (mu_R.+pp.*dl.-(rho.-l).*pp)./zp
    dnn .= (mu_R.-nn.*dl.-(rho.+l).*nn)./zn
    dzp .= (mu_R.-zp.*dpp)./pp.-zp
    dzn .= (mu_R.-zn.*dnn)./nn.-zn
end

# Kernel functions ---------------------------------------------------------
is_valid(val::Real) = !(isnan(val) || isinf(val))
function is_valid(vec::AbstractArray)
    @inbounds for i=1:length(vec)
        is_valid(vec[i]) || return false
    end
    return true
end
is_valid(args...) = all(is_valid(arg) for arg in args)

function get_varphi(obj_val,x_lr,xl_r,xu_r,x_ur,mu)
    varphi = obj_val
    @simd for i=1:length(x_lr)
        @inbounds xll = x_lr[i]-xl_r[i]
        xll < 0 && return Inf
        varphi -= mu*log(xll)
    end
    @simd for i=1:length(x_ur)
        @inbounds xuu = xu_r[i]-x_ur[i]
        xuu < 0 && return Inf
        varphi -= mu*log(xuu)
    end
    return varphi
end
get_inf_pr(c) = norm(c,Inf)
function get_inf_du(f,zl,zu,jacl,sd)
    inf_du = 0.
    @simd for i=1:length(f)
        @inbounds inf_du = max(inf_du,abs(f[i]-zl[i]+zu[i]+jacl[i]))
    end
    return inf_du/sd
end
function get_inf_compl(x_lr,xl_r,zl_r,xu_r,x_ur,zu_r,mu,sc)
    inf_compl = 0.
    @simd for i=1:length(x_lr)
        @inbounds inf_compl = max(inf_compl,abs((x_lr[i]-xl_r[i])*zl_r[i]-mu))
    end
    @simd for i=1:length(x_ur)
        @inbounds inf_compl = max(inf_compl,abs((xu_r[i]-x_ur[i])*zu_r[i]-mu))
    end
    return inf_compl/sc
end
function get_varphi_d(f,x,xl,xu,dx,mu)
    varphi_d = 0.
    @simd for i=1:length(f)
        @inbounds varphi_d += (f[i] - mu/(x[i]-xl[i]) + mu/(xu[i]-x[i])) *dx[i]
    end
    return varphi_d
end
function get_alpha_max(x,xl,xu,dx,tau)
    alpha_max = 1.
    @simd for i=1:length(x)
        @inbounds dx[i]<0 && (alpha_max=min(alpha_max,(-x[i]+xl[i])*tau/dx[i]))
        @inbounds dx[i]>0 && (alpha_max=min(alpha_max,(-x[i]+xu[i])*tau/dx[i]))
    end
    return alpha_max
end
function get_alpha_z(zl_r,zu_r,dzl,dzu,tau)
    alpha_z = 1.
    @simd for i=1:length(zl_r)
        @inbounds dzl[i]<0 && (alpha_z=min(alpha_z,-zl_r[i]*tau/dzl[i]))
     end
    @simd for i=1:length(zu_r)
        @inbounds dzu[i]<0 && (alpha_z=min(alpha_z,-zu_r[i]*tau/dzu[i]))
    end
    return alpha_z
end
function get_obj_val_R(p,n,D_R,x,x_ref,rho,zeta)
    obj_val_R = 0.
    @simd for i=1:length(p)
        @inbounds obj_val_R += rho*(p[i]+n[i]) .+ zeta/2*D_R[i]^2*(x[i]-x_ref[i])^2
    end
    return obj_val_R
end
get_theta(c) = norm(c,1)
function get_theta_R(c,p,n)
    theta_R = 0.
    @simd for i=1:length(c)
        @inbounds theta_R += abs(c[i]-p[i]+n[i])
    end
    return theta_R
end
function get_inf_pr_R(c,p,n)
    inf_pr_R = 0.
    @simd for i=1:length(c)
        @inbounds inf_pr_R = max(inf_pr_R,abs(c[i]-p[i]+n[i]))
    end
    return inf_pr_R
end
function get_inf_du_R(f_R,l,zl,zu,jacl,zp,zn,rho,sd)
    inf_du_R = 0.
    @simd for i=1:length(zl)
        @inbounds inf_du_R = max(inf_du_R,abs(f_R[i]-zl[i]+zu[i]+jacl[i]))
    end
    @simd for i=1:length(zp)
        @inbounds inf_du_R = max(inf_du_R,abs(rho-l[i]-zp[i]))
    end
    @simd for i=1:length(zn)
        @inbounds inf_du_R = max(inf_du_R,abs(rho+l[i]-zn[i]))
    end
    return inf_du_R/sd
end
function get_inf_compl_R(x_lr,xl_r,zl_r,xu_r,x_ur,zu_r,pp,zp,nn,zn,mu_R,sc)
    inf_compl_R = 0.
    @simd for i=1:length(x_lr)
        @inbounds inf_compl_R = max(inf_compl_R,abs((x_lr[i]-xl_r[i])*zl_r[i]-mu_R))
    end
    @simd for i=1:length(xu_r)
        @inbounds inf_compl_R = max(inf_compl_R,abs((xu_r[i]-x_ur[i])*zu_r[i]-mu_R))
    end
    @simd for i=1:length(pp)
        @inbounds inf_compl_R = max(inf_compl_R,abs(pp[i]*zp[i]-mu_R))
    end
    @simd for i=1:length(nn)
        @inbounds inf_compl_R = max(inf_compl_R,abs(nn[i]*zn[i]-mu_R))
    end
    return inf_compl_R/sc
end
function get_alpha_max_R(x,xl,xu,dx,pp,dpp,nn,dnn,tau_R)
    alpha_max_R = 1.
    @simd for i=1:length(x)
        @inbounds dx[i]<0 && (alpha_max_R=min(alpha_max_R,(-x[i]+xl[i])*tau_R/dx[i]))
        @inbounds dx[i]>0 && (alpha_max_R=min(alpha_max_R,(-x[i]+xu[i])*tau_R/dx[i]))
    end
    @simd for i=1:length(pp)
        @inbounds dpp[i]<0 && (alpha_max_R=min(alpha_max_R,-pp[i]*tau_R/dpp[i]))
    end
    @simd for i=1:length(nn)
        @inbounds dnn[i]<0 && (alpha_max_R=min(alpha_max_R,-nn[i]*tau_R/dnn[i]))
    end
    return alpha_max_R
end
function get_alpha_z_R(zl_r,zu_r,dzl,dzu,zp,dzp,zn,dzn,tau_R)
    alpha_z_R = 1.
    @simd for i=1:length(zl_r)
        @inbounds dzl[i]<0 && (alpha_z_R=min(alpha_z_R,-zl_r[i]*tau_R/dzl[i]))
    end
    @simd for i=1:length(zu_r)
        @inbounds dzu[i]<0 && (alpha_z_R=min(alpha_z_R,-zu_r[i]*tau_R/dzu[i]))
    end
    @simd for i=1:length(zp)
        @inbounds dzp[i]<0 && (alpha_z_R=min(alpha_z_R,-zp[i]*tau_R/dzp[i]))
    end
    @simd for i=1:length(zn)
        @inbounds dzn[i]<0 && (alpha_z_R=min(alpha_z_R,-zn[i]*tau_R/dzn[i]))
    end
    return alpha_z_R
end
function get_varphi_R(obj_val,x_lr,xl_r,xu_r,x_ur,pp,nn,mu_R)
    varphi_R = obj_val
    @simd for i=1:length(x_lr)
        @inbounds xll = x_lr[i]-xl_r[i]
        xll < 0 && return Inf
        varphi_R -= mu_R*log(xll)
    end
    @simd for i=1:length(x_ur)
        @inbounds xuu = xu_r[i]-x_ur[i]
        xuu < 0 && return Inf
        varphi_R -= mu_R*log(xuu)
    end
    @simd for i=1:length(pp)
        @inbounds pp[i] < 0 && return Inf
        @inbounds varphi_R -= mu_R*log(pp[i])
    end
    @simd for i=1:length(pp)
        @inbounds nn[i] < 0 && return Inf
        @inbounds varphi_R -= mu_R*log(nn[i])
    end
    return varphi_R
end
function get_F(c,f,zl,zu,jacl,x_lr,xl_r,zl_r,xu_r,x_ur,zu_r,mu)
    F = 0.
    for i=1:length(c)
        @inbounds F = max(F,c[i])
    end
    for i=1:length(f)
        @inbounds F = max(F,f[i]-zl[i]+zu[i]+jacl[i])
    end
    for i=1:length(x_lr)
        x_lr[i] >= xl_r[i] || return Inf
        zl_r[i] >= 0       || return Inf
        @inbounds F = max(F,(x_lr[i]-xl_r[i])*zl_r[i]-mu)
    end
    for i=1:length(x_ur)
        xu_r[i] >= x_ur[i] || return Inf
        zu_r[i] >= 0       || return Inf
        @inbounds F = max(F,(xu_r[i]-xu_r[i])*zu_r[i]-mu)
    end
    return F
end
function get_varphi_d_R(f_R,x,xl,xu,dx,pp,nn,dpp,dnn,mu_R,rho)
    varphi_d = 0.
    @simd for i=1:length(x)
        @inbounds varphi_d += (f_R[i] - mu_R/(x[i]-xl[i]) + mu_R/(xu[i]-x[i])) *dx[i]
    end
    @simd for i=1:length(pp)
        @inbounds varphi_d += (rho - mu_R/pp[i]) *dpp[i]
    end
    @simd for i=1:length(nn)
        @inbounds varphi_d += (rho - mu_R/nn[i]) *dnn[i]
    end
    return varphi_d
end
function initialize_variables!(x,xl,xu,bound_push,bound_fac)
    @inbounds @simd for i=1:length(x)
        if xl[i]!=-Inf && xu[i]!=Inf
            x[i]=min(xu[i]-min(bound_push*max(1,abs(xu[i])),bound_fac*(xu[i]-xl[i])),
                     max(xl[i]+min(bound_push*max(1,abs(xl[i])),bound_fac*(xu[i]-xl[i])),x[i]))
        elseif xl[i]!=-Inf && xu[i]==Inf
            x[i]=max(xl[i]+bound_push*max(1,abs(xl[i])),x[i])
        elseif xl[i]==-Inf && xu[i]!=Inf
            x[i]=min(xu[i]-bound_push*max(1,abs(xu[i])),x[i])
        end
    end
end

function set_con_scale!(con_scale::AbstractVector, jac::SparseMatrixCOO, nlp_scaling_max_gradient)
    @simd for i in 1:nnz(jac)
        row = @inbounds jac.I[i]
        @inbounds con_scale[row] = max(con_scale[row], abs(jac.V[i]))
    end
    con_scale .= min.(1.0, nlp_scaling_max_gradient ./ con_scale)
end
function set_con_scale!(con_scale::AbstractVector, jac::Matrix, nlp_scaling_max_gradient)
    for row in 1:size(jac, 1)
        for col in 1:size(jac, 2)
            @inbounds con_scale[row] = max(con_scale[row], abs(jac[row, col]))
        end
    end
    con_scale .= min.(1.0, nlp_scaling_max_gradient ./ con_scale)
end

function adjust_boundary!(x_lr,xl_r,x_ur,xu_r,mu)
    adjusted = 0
    c1 = eps(Float64)*mu
    c2= eps(Float64)^(3/4)
    @simd for i=1:length(xl_r)
        @inbounds x_lr[i]-xl_r[i] < c1 && (xl_r[i] -= c2*max(1,abs(x_lr[i]));adjusted+=1)
    end
    @simd for i=1:length(xu_r)
        @inbounds xu_r[i]-x_ur[i] < c1 && (xu_r[i] += c2*max(1,abs(x_ur[i]));adjusted+=1)
    end
    return adjusted
end
function get_rel_search_norm(x,dx)
    rel_search_norm = 0.
    @simd for i=1:length(x)
        @inbounds rel_search_norm = max(rel_search_norm,abs(dx[i])/(1. +abs(x[i])))
    end
    return rel_search_norm
end
function force_lower_triangular!(I,J)
    @simd for i=1:length(I)
        @inbounds if J[i] > I[i]
            tmp=J[i]
            J[i]=I[i]
            I[i]=tmp
        end
    end
end

# Utility functions
get_sd(l,zl_r,zu_r,s_max) =
    max(s_max,(norm(l,1)+norm(zl_r,1)+norm(zu_r,1)) / max(1,(length(l)+length(zl_r)+length(zu_r))))/s_max
get_sc(zl_r,zu_r,s_max) =
    max(s_max,(norm(zl_r,1)+norm(zu_r,1)) / max(1,length(zl_r)+length(zu_r)))/s_max
get_mu(mu,mu_min,mu_linear_decrease_factor,mu_superlinear_decrease_power,tol) =
    max(mu_min, max(tol/10,min(mu_linear_decrease_factor*mu,mu^mu_superlinear_decrease_power)))
get_tau(mu,tau_min)=max(tau_min,1-mu)
function get_alpha_min(theta,varphi_d,theta_min,gamma_theta,gamma_phi,alpha_min_frac,del,s_theta,s_phi)
    if varphi_d<0
        if theta<=theta_min
            return alpha_min_frac*min(
                gamma_theta,gamma_phi*theta/(-varphi_d),
                del*theta^s_theta/(-varphi_d)^s_phi)
        else
            return alpha_min_frac*min(
                gamma_theta,gamma_phi*theta/(-varphi_d))
        end
    else
        return alpha_min_frac*gamma_theta
    end
end
is_switching(varphi_d,alpha,s_phi,del,theta,s_theta) = varphi_d < 0 && alpha*(-varphi_d)^s_phi > del*theta^s_theta
is_armijo(varphi_trial,varphi,eta_phi,alpha,varphi_d) = varphi_trial <= varphi + eta_phi*alpha*varphi_d
is_sufficient_progress(theta_trial,theta,gamma_theta,varphi_trial,varphi,gamma_phi,has_constraints) =
    (has_constraints && ((theta_trial<=(1-gamma_theta)*theta+10*eps(Float64)*abs(theta))) ||
    ((varphi_trial<=varphi-gamma_phi*theta +10*eps(Float64)*abs(varphi))))
augment_filter!(filter,theta,varphi,gamma_theta) = push!(filter,((1-gamma_theta)*theta,varphi-gamma_theta*theta))
function is_filter_acceptable(filter,theta,varphi)
    !isnan(theta) || return false
    !isinf(theta) || return false
    !isnan(varphi) || return false
    !isinf(varphi) || return false

    for (theta_F,varphi_F) in filter
        theta <= theta_F || varphi <= varphi_F || return false
    end
    return true
end
is_barr_obj_rapid_increase(varphi,varphi_trial,obj_max_inc) =
    varphi_trial >= varphi && log(10,varphi_trial-varphi) > obj_max_inc + max(1.,log(10,abs(varphi)))
reset_bound_dual!(z,x,mu,kappa_sigma) = (z.=max.(min.(z,kappa_sigma.*mu./x),mu/kappa_sigma./x))
reset_bound_dual!(z,x1,x2,mu,kappa_sigma) = (z.=max.(min.(z,(kappa_sigma*mu)./(x1.-x2)),(mu/kappa_sigma)./(x1.-x2)))
function get_ftype(filter,theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
                   theta_min,obj_max_inc,gamma_theta,gamma_phi,has_constraints)
    is_filter_acceptable(filter,theta_trial,varphi_trial) || return " "
    !is_barr_obj_rapid_increase(varphi,varphi_trial,obj_max_inc) || return " "

    if theta <= theta_min && switching_condition
        armijo_condition && return "f"
    else
        is_sufficient_progress(
            theta_trial,theta,gamma_theta,varphi_trial,varphi,gamma_phi,has_constraints) && return "h"
    end

    return " "
end

# fixed variable treatment ----------------------------------------------------
function _get_fixed_variable_index(
    mat::SparseMatrixCSC{Tv,Ti1}, ind_fixed::Vector{Ti2}) where {Tv,Ti1,Ti2}

    fixed_aug_index = Int[]
    for i in ind_fixed
        append!(fixed_aug_index,append!(collect(mat.colptr[i]+1:mat.colptr[i+1]-1)))
    end
    append!(fixed_aug_index,setdiff!(Base._findin(mat.rowval,ind_fixed),mat.colptr))

    return fixed_aug_index
end
fixed_variable_treatment_vec!(vec,ind_fixed) = (vec[ind_fixed] .= 0.)
function fixed_variable_treatment_z!(zl,zu,f,jacl,ind_fixed)
    @simd for i in ind_fixed
        z = f[i]+jacl[i]
        z >=0 ? (zl[i] = z; zu[i] = 0.) : (zl[i] = 0.; zu[i] = -z)
    end
end

function dual_inf_perturbation!(px,ind_llb,ind_uub,mu,kappa_d)
    @simd for i in ind_llb
        @inbounds px[i] -= mu*kappa_d
    end
    @simd for i in ind_uub
        @inbounds px[i] += mu*kappa_d
    end
end

# Print functions -----------------------------------------------------------
function print_init(ips::AbstractInteriorPointSolver)
    @notice(ips.logger,@sprintf("Number of nonzeros in constraint Jacobian............: %8i",get_nnzj(ips.nlp)))
    @notice(ips.logger,@sprintf("Number of nonzeros in Lagrangian Hessian.............: %8i\n",get_nnzh(ips.nlp)))

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
        ips.cnt.k == 0 ? 0. : norm(ips.dx,Inf),
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
                number of nonzeros in lagrangian hessian.: $(get_nnzh(ips.nlp))
                number of nonzeros in constraint jacobian: $(get_nnzj(ips.nlp))
                status...................................: $(ips.status)
                """
end
print(io::IO,ips::AbstractInteriorPointSolver) = print(io, string(ips))
show(io::IO,ips::AbstractInteriorPointSolver) = print(io,ips)

function get_index_constraints(nlp::AbstractNLPModel; fixed_variable_treatment=MAKE_PARAMETER)
    ind_ineq = findall(get_lcon(nlp) .!= get_ucon(nlp))
    xl = [get_lvar(nlp);view(get_lcon(nlp),ind_ineq)]
    xu = [get_uvar(nlp);view(get_ucon(nlp),ind_ineq)]
    if fixed_variable_treatment == MAKE_PARAMETER
        ind_fixed = findall(xl .== xu)
        ind_lb = findall((xl .!= -Inf) .* (xl .!= xu))
        ind_ub = findall((xu .!=  Inf) .* (xl .!= xu))
    else
        ind_fixed = Int[]
        ind_lb = findall(xl .!=-Inf)
        ind_ub = findall(xu .!= Inf)
    end

    ind_llb = findall((get_lvar(nlp) .== -Inf).*(get_uvar(nlp) .!= Inf))
    ind_uub = findall((get_lvar(nlp) .!= -Inf).*(get_uvar(nlp) .== Inf))

    # Return named tuple
    return (
        ind_ineq=ind_ineq,
        ind_fixed=ind_fixed,
        ind_lb=ind_lb,
        ind_ub=ind_ub,
        ind_llb=ind_llb,
        ind_uub=ind_uub,
    )
end

function madnlp(model::AbstractNLPModel;buffered=true, kwargs...)
    ips = InteriorPointSolver(model;kwargs...)
    initialize!(ips.kkt)
    return optimize!(ips)
end

# Utilities
has_constraints(ips) = ips.m != 0

