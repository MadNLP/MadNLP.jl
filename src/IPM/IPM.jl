# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

abstract type AbstractInteriorPointSolver{T} end

include("restoration.jl")

mutable struct InteriorPointSolver{T,KKTSystem <: AbstractKKTSystem{T}} <: AbstractInteriorPointSolver{T}
    nlp::AbstractNLPModel
    kkt::KKTSystem

    opt::Options
    cnt::Counters
    logger::Logger

    n::Int # number of variables (after reformulation)
    m::Int # number of cons
    nlb::Int
    nub::Int

    x::Vector{T} # primal (after reformulation)
    l::Vector{T} # dual
    zl::Vector{T} # dual (after reformulation)
    zu::Vector{T} # dual (after reformulation)
    xl::Vector{T} # primal lower bound (after reformulation)
    xu::Vector{T} # primal upper bound (after reformulation)

    obj_val::T
    f::Vector{T}
    c::Vector{T}

    jacl::Vector{T}

    d::Vector{T}
    dx::StrideOneVector{T}
    dl::StrideOneVector{T}
    dzl::StrideOneVector{T}
    dzu::StrideOneVector{T}

    p::Vector{T}
    px::StrideOneVector{T}
    pl::StrideOneVector{T}

    pzl::Union{Nothing,StrideOneVector{T}}
    pzu::Union{Nothing,StrideOneVector{T}}

    _w1::Vector{T}
    _w1x::StrideOneVector{T}
    _w1l::StrideOneVector{T}
    _w1zl::Union{Nothing,StrideOneVector{T}}
    _w1zu::Union{Nothing,StrideOneVector{T}}

    _w2::Vector{T}
    _w2x::StrideOneVector{T}
    _w2l::StrideOneVector{T}
    _w2zl::Union{Nothing,StrideOneVector{T}}
    _w2zu::Union{Nothing,StrideOneVector{T}}

    _w3::Vector{T}
    _w3x::StrideOneVector{T}
    _w3l::StrideOneVector{T}

    _w4::Vector{T}
    _w4x::StrideOneVector{T}
    _w4l::StrideOneVector{T}

    x_trial::Vector{T}
    c_trial::Vector{T}
    obj_val_trial::T

    x_slk::StrideOneVector{T}
    c_slk::SubVector{T}
    rhs::Vector{T}

    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    ind_llb::Vector{Int}
    ind_uub::Vector{Int}

    x_lr::SubVector{T}
    x_ur::SubVector{T}
    xl_r::SubVector{T}
    xu_r::SubVector{T}
    zl_r::SubVector{T}
    zu_r::SubVector{T}

    dx_lr::SubVector{T}
    dx_ur::SubVector{T}
    x_trial_lr::SubVector{T}
    x_trial_ur::SubVector{T}

    linear_solver::AbstractLinearSolver
    iterator::AbstractIterator

    obj_scale::Vector{T}
    con_scale::Vector{T}
    con_jac_scale::Vector{T}
    inf_pr::T
    inf_du::T
    inf_compl::T

    theta_min::T
    theta_max::T
    mu::T
    tau::T

    alpha::T
    alpha_z::T
    ftype::String

    del_w::T
    del_c::T
    del_w_last::T

    filter::Vector{Tuple{T,T}}

    RR::Union{Nothing,RobustRestorer}
    status::Status
    output::Dict
end

function InteriorPointSolver(nlp::AbstractNLPModel{T};
    option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(), kwargs...
) where T
    opt = Options(linear_solver=default_linear_solver())
    set_options!(opt,option_dict,kwargs)
    check_option_sanity(opt)

    KKTSystem = if opt.kkt_system == SPARSE_KKT_SYSTEM
        MT = (opt.linear_solver.INPUT_MATRIX_TYPE == :csc) ? SparseMatrixCSC{T, Int32} : Matrix{T}
        SparseKKTSystem{T, MT}
    elseif opt.kkt_system == SPARSE_UNREDUCED_KKT_SYSTEM
        MT = (opt.linear_solver.INPUT_MATRIX_TYPE == :csc) ? SparseMatrixCSC{T, Int32} : Matrix{T}
        SparseUnreducedKKTSystem{T, MT}
    elseif opt.kkt_system == DENSE_KKT_SYSTEM
        MT = Matrix{T}
        VT = Vector{T}
        DenseKKTSystem{T, VT, MT}
    elseif opt.kkt_system == DENSE_CONDENSED_KKT_SYSTEM
        MT = Matrix{T}
        VT = Vector{T}
        DenseCondensedKKTSystem{T, VT, MT}
    end
    return InteriorPointSolver{T,KKTSystem}(nlp, opt; option_linear_solver=option_dict)
end

# Inner constructor
function InteriorPointSolver{T,KKTSystem}(nlp::AbstractNLPModel, opt::Options;
    option_linear_solver::Dict{Symbol,Any}=Dict{Symbol,Any}(),
) where {T, KKTSystem<:AbstractKKTSystem{T}}
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
    x = [get_x0(nlp);zeros(T,ns)]
    l = copy(get_y0(nlp))
    zl= zeros(T,get_nvar(nlp)+ns)
    zu= zeros(T,get_nvar(nlp)+ns)

    f = zeros(T,n) # not sure why, but seems necessary to initialize to 0 when used with Plasmo interface
    c = zeros(T,m)

    n_jac = nnz_jacobian(kkt)

    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    x_trial=Vector{T}(undef,n)
    c_trial=Vector{T}(undef,m)

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

    _w1 = zeros(T,aug_vec_length) # fixes the random failure for inertia-free + Unreduced
    _w1x= view(_w1,1:n)
    _w1l= view(_w1,n+1:n+m)
    _w1zl = is_reduced(kkt) ? nothing : view(_w1,n+m+1:n+m+nlb)
    _w1zu = is_reduced(kkt) ? nothing : view(_w1,n+m+nlb+1:n+m+nlb+nub)


    _w2 = Vector{T}(undef,aug_vec_length)
    _w2x= view(_w2,1:n)
    _w2l= view(_w2,n+1:n+m)
    _w2zl = is_reduced(kkt) ? nothing : view(_w2,n+m+1:n+m+nlb)
    _w2zu = is_reduced(kkt) ? nothing : view(_w2,n+m+nlb+1:n+m+nlb+nub)

    _w3 = zeros(T,aug_vec_length) # fixes the random failure for inertia-free + Unreduced
    _w3x= view(_w3,1:n)
    _w3l= view(_w3,n+1:n+m)
    _w4 = zeros(T,aug_vec_length) # need to initialize to zero due to mul!
    _w4x= view(_w4,1:n)
    _w4l= view(_w4,n+1:n+m)


    jacl = zeros(T,n) # spblas may throw an error if not initialized to zero

    d = Vector{T}(undef,aug_vec_length)
    dx= view(d,1:n)
    dl= view(d,n+1:n+m)
    dzl= is_reduced(kkt) ? Vector{T}(undef,nlb) : view(d,n+m+1:n+m+nlb)
    dzu= is_reduced(kkt) ? Vector{T}(undef,nub) : view(d,n+m+nlb+1:n+m+nlb+nub)
    dx_lr = view(dx,ind_cons.ind_lb)
    dx_ur = view(dx,ind_cons.ind_ub)

    p = Vector{T}(undef,aug_vec_length)
    px= view(p,1:n)
    pl= view(p,n+1:n+m)
    pzl= is_reduced(kkt) ? Vector{T}(undef,nlb) : view(p,n+m+1:n+m+nlb)
    pzu= is_reduced(kkt) ? Vector{T}(undef,nub) : view(p,n+m+nlb+1:n+m+nlb+nub)

    obj_scale = [1.0]
    con_scale = ones(T,m)
    con_jac_scale = ones(T,n_jac)

    @trace(logger,"Initializing linear solver.")
    cnt.linear_solver_time =
        @elapsed linear_solver = opt.linear_solver.Solver(get_kkt(kkt) ; option_dict=option_linear_solver,logger=logger)

    n_kkt = size(get_kkt(kkt), 1)
    @trace(logger,"Initializing iterative solver.")
    iterator = opt.iterator.Solver(
        similar(d, n_kkt),
        (b, x)->mul!(b, kkt, x), (x)->solve!(linear_solver, x) ; option_dict=option_linear_solver)

    @trace(logger,"Initializing fixed variable treatment scheme.")

    if opt.inertia_correction_method == INERTIA_AUTO
        opt.inertia_correction_method = is_inertia(linear_solver) ? INERTIA_BASED : INERTIA_FREE
    end

    !isempty(option_linear_solver) && print_ignored_options(logger, option_linear_solver)

    return InteriorPointSolver{T,KKTSystem}(nlp,kkt,opt,cnt,logger,
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
        Vector{T}[],nothing,INITIAL,Dict(),
    )
end

include("utils.jl")
include("kernels.jl")
include("callbacks.jl")
include("factorization.jl")
include("solver.jl")

