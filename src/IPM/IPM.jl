# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

abstract type AbstractInteriorPointSolver end

include("restoration.jl")

"""
TODO
"""
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

    d::AbstractKKTVector{Float64, Vector{Float64}}
    p::AbstractKKTVector{Float64, Vector{Float64}}

    _w1::AbstractKKTVector{Float64, Vector{Float64}}
    _w2::AbstractKKTVector{Float64, Vector{Float64}}

    _w3::AbstractKKTVector{Float64, Vector{Float64}}
    _w4::AbstractKKTVector{Float64, Vector{Float64}}

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

"""
TODO
"""
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
    elseif opt.kkt_system == DENSE_CONDENSED_KKT_SYSTEM
        MT = Matrix{Float64}
        VT = Vector{Float64}
        DenseCondensedKKTSystem{Float64, VT, MT}
    end
    return InteriorPointSolver{KKTSystem}(nlp, opt; option_linear_solver=option_dict)
end

# Inner constructor
"""
TODO
"""
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

    _w1 = is_reduced(kkt) ? ReducedKKTVector(n, m) : UnreducedKKTVector(n, m, nlb, nub)
    _w2 = is_reduced(kkt) ? ReducedKKTVector(n, m) : UnreducedKKTVector(n, m, nlb, nub)
    _w3 = is_reduced(kkt) ? ReducedKKTVector(n, m) : UnreducedKKTVector(n, m, nlb, nub)
    _w4 = is_reduced(kkt) ? ReducedKKTVector(n, m) : UnreducedKKTVector(n, m, nlb, nub)

    jacl = zeros(n) # spblas may throw an error if not initialized to zero

    d = UnreducedKKTVector(n, m, nlb, nub)
    dx_lr = view(d.xp, ind_cons.ind_lb) # TODO
    dx_ur = view(d.xp, ind_cons.ind_ub) # TODO

    p = UnreducedKKTVector(n, m, nlb, nub)

    obj_scale = [1.0]
    con_scale = ones(m)
    con_jac_scale = ones(n_jac)

    @trace(logger,"Initializing linear solver.")
    cnt.linear_solver_time =
        @elapsed linear_solver = opt.linear_solver.Solver(get_kkt(kkt) ; option_dict=option_linear_solver,logger=logger)

    n_kkt = size(kkt, 1)
    buffer_vec = similar(full(d), n_kkt)
    @trace(logger,"Initializing iterative solver.")
    iterator = opt.iterator(linear_solver, kkt, buffer_vec)

    @trace(logger,"Initializing fixed variable treatment scheme.")

    if opt.inertia_correction_method == INERTIA_AUTO
        opt.inertia_correction_method = is_inertia(linear_solver) ? INERTIA_BASED : INERTIA_FREE
    end

    !isempty(option_linear_solver) && print_ignored_options(logger, option_linear_solver)

    return InteriorPointSolver{KKTSystem}(nlp,kkt,opt,cnt,logger,
        n,m,nlb,nub,x,l,zl,zu,xl,xu,0.,f,c,
        jacl,
        d, p,
        _w1, _w2, _w3, _w4,
        x_trial,c_trial,0.,x_slk,c_slk,rhs,
        ind_cons.ind_ineq,ind_cons.ind_fixed,ind_cons.ind_llb,ind_cons.ind_uub,
        x_lr,x_ur,xl_r,xu_r,zl_r,zu_r,dx_lr,dx_ur,x_trial_lr,x_trial_ur,
        linear_solver,iterator,
        obj_scale,con_scale,con_jac_scale,
        0.,0.,0.,0.,0.,0.,0.,0.,0.," ",0.,0.,0.,
        Vector{Float64}[],nothing,INITIAL,Dict(),
    )
end

include("utils.jl")
include("kernels.jl")
include("callbacks.jl")
include("factorization.jl")
include("solver.jl")

