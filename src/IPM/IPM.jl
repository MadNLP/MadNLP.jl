# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

abstract type AbstractMadNLPSolver{T} end

include("restoration.jl")

mutable struct MadNLPSolver{T, KKTSystem <: AbstractKKTSystem{T}, Model <: AbstractNLPModel, LinSolver <: AbstractLinearSolver{T}, Iterator <: AbstractIterator{T}, KKTVec <: AbstractKKTVector{T, Vector{T}}} <: AbstractMadNLPSolver{T}
    nlp::Model
    kkt::KKTSystem

    opt::MadNLPOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger

    n::Int # number of variables (after reformulation)
    m::Int # number of cons
    nlb::Int
    nub::Int

    x::PrimalVector{T, Vector{T}} # primal (after reformulation)
    y::Vector{T} # dual
    zl::PrimalVector{T, Vector{T}} # dual (after reformulation)
    zu::PrimalVector{T, Vector{T}} # dual (after reformulation)
    xl::PrimalVector{T, Vector{T}} # primal lower bound (after reformulation)
    xu::PrimalVector{T, Vector{T}} # primal upper bound (after reformulation)

    obj_val::T
    f::PrimalVector{T, Vector{T}}
    c::Vector{T}

    jacl::Vector{T}

    d::UnreducedKKTVector{T, Vector{T}}
    p::UnreducedKKTVector{T, Vector{T}}

    _w1::KKTVec
    _w2::KKTVec

    _w3::KKTVec
    _w4::KKTVec

    x_trial::PrimalVector{T, Vector{T}}
    c_trial::Vector{T}
    obj_val_trial::T

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

    linear_solver::LinSolver
    iterator::Iterator

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

    RR::Union{Nothing,RobustRestorer{T}}
    status::Status
    output::Dict
end

function MadNLPSolver(nlp::AbstractNLPModel{T}; kwargs...) where T
    opt_ipm, opt_linear_solver, logger = load_options(; kwargs...)
    @assert is_supported(opt_ipm.linear_solver, T)

    VT = Vector{T}
    # Determine Hessian approximation
    QN = if opt_ipm.hessian_approximation == DENSE_BFGS
        BFGS{T, VT}
    elseif opt_ipm.hessian_approximation == DENSE_DAMPED_BFGS
        DampedBFGS{T, VT}
    elseif opt_ipm.hessian_approximation == SPARSE_COMPACT_LBFGS
        CompactLBFGS{T, VT, Matrix{T}}
    else
        ExactHessian{T, VT}
    end
    # Determine KKT system
    KKTSystem = if opt_ipm.kkt_system == SPARSE_KKT_SYSTEM
        MT = (input_type(opt_ipm.linear_solver) == :csc) ? SparseMatrixCSC{T, Int32} : Matrix{T}
        SparseKKTSystem{T, VT, MT, QN}
    elseif opt_ipm.kkt_system == SPARSE_CONDENSED_KKT_SYSTEM
        MT = (input_type(opt_ipm.linear_solver) == :csc) ? SparseMatrixCSC{T, Int32} : Matrix{T}
        SparseCondensedKKTSystem{T, VT, MT, QN}
    elseif opt_ipm.kkt_system == SPARSE_UNREDUCED_KKT_SYSTEM
        MT = (input_type(opt_ipm.linear_solver) == :csc) ? SparseMatrixCSC{T, Int32} : Matrix{T}
        SparseUnreducedKKTSystem{T, VT, MT, QN}
    elseif opt_ipm.kkt_system == DENSE_KKT_SYSTEM
        MT = Matrix{T}
        DenseKKTSystem{T, VT, MT, QN}
    elseif opt_ipm.kkt_system == DENSE_CONDENSED_KKT_SYSTEM
        MT = Matrix{T}
        DenseCondensedKKTSystem{T, VT, MT, QN}
    end
    return MadNLPSolver{T,KKTSystem}(nlp, opt_ipm, opt_linear_solver; logger=logger)
end

# Constructor for unregistered KKT systems
function MadNLPSolver{T, KKTSystem}(nlp::AbstractNLPModel{T}; options...) where {T, KKTSystem}
    opt_ipm, opt_linear_solver, logger = load_options(; options...)
    @assert is_supported(opt_ipm.linear_solver, T)
    return MadNLPSolver{T,KKTSystem}(nlp, opt_ipm, opt_linear_solver; logger=logger)
end

# Inner constructor
function MadNLPSolver{T,KKTSystem}(
    nlp::AbstractNLPModel,
    opt::MadNLPOptions,
    opt_linear_solver::AbstractOptions;
    logger=MadNLPLogger(),
) where {T, KKTSystem<:AbstractKKTSystem{T}}
    cnt = MadNLPCounters(start_time=time())

    # generic options
    opt.disable_garbage_collector &&
        (GC.enable(false); @warn(logger,"Julia garbage collector is temporarily disabled"))
    set_blas_num_threads(opt.blas_num_threads; permanent=true)
    @trace(logger,"Initializing variables.")
    ind_cons = get_index_constraints(nlp; fixed_variable_treatment=opt.fixed_variable_treatment)
    ns = length(ind_cons.ind_ineq)
    nx = get_nvar(nlp)
    n = nx+ns
    m = get_ncon(nlp)

    # Initialize KKT
    kkt = KKTSystem(nlp, ind_cons)

    # Primal variable
    x = PrimalVector{T, Vector{T}}(nx, ns, ind_cons)
    variable(x) .= get_x0(nlp)
    # Bounds
    xl = PrimalVector{T, Vector{T}}(nx, ns, ind_cons)
    variable(xl) .= get_lvar(nlp)
    slack(xl) .= view(get_lcon(nlp), ind_cons.ind_ineq)
    xu = PrimalVector{T, Vector{T}}(nx, ns, ind_cons)
    variable(xu) .= get_uvar(nlp)
    slack(xu) .= view(get_ucon(nlp), ind_cons.ind_ineq)
    zl = PrimalVector{T, Vector{T}}(nx, ns, ind_cons)
    zu = PrimalVector{T, Vector{T}}(nx, ns, ind_cons)
    
    # Gradient
    f = PrimalVector{T, Vector{T}}(nx, ns, ind_cons)

    y = copy(get_y0(nlp))
    c = zeros(T, m)

    n_jac = nnz_jacobian(kkt)

    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    x_trial = PrimalVector{T, Vector{T}}(nx, ns, ind_cons)
    c_trial = Vector{T}(undef, m)

    c_slk = view(c,ind_cons.ind_ineq)
    rhs = (get_lcon(nlp).==get_ucon(nlp)).*get_lcon(nlp)

    x_lr = view(full(x), ind_cons.ind_lb)
    x_ur = view(full(x), ind_cons.ind_ub)
    xl_r = view(full(xl), ind_cons.ind_lb)
    xu_r = view(full(xu), ind_cons.ind_ub)
    zl_r = view(full(zl), ind_cons.ind_lb)
    zu_r = view(full(zu), ind_cons.ind_ub)
    x_trial_lr = view(full(x_trial), ind_cons.ind_lb)
    x_trial_ur = view(full(x_trial), ind_cons.ind_ub)

    
    # if is_reduced(kkt)
    #     _w1 = ReducedKKTVector{T,typeof(c)}(n, m, ind_cons)
    #     _w2 = ReducedKKTVector{T,typeof(c)}(n, m, ind_cons)
    #     _w3 = ReducedKKTVector{T,typeof(c)}(n, m, ind_cons)
    #     _w4 = ReducedKKTVector{T,typeof(c)}(n, m, ind_cons)
    # else
    _w1 = UnreducedKKTVector{T,typeof(c)}(n, m, nlb, nub, ind_cons)
    _w2 = UnreducedKKTVector{T,typeof(c)}(n, m, nlb, nub, ind_cons)
    _w3 = UnreducedKKTVector{T,typeof(c)}(n, m, nlb, nub, ind_cons)
    _w4 = UnreducedKKTVector{T,typeof(c)}(n, m, nlb, nub, ind_cons)
    # end

    jacl = zeros(T,n) # spblas may throw an error if not initialized to zero

    d = UnreducedKKTVector{T,typeof(c)}(n, m, nlb, nub, ind_cons)
    dx_lr = view(d.xp, ind_cons.ind_lb) # TODO
    dx_ur = view(d.xp, ind_cons.ind_ub) # TODO

    p = UnreducedKKTVector{T,typeof(c)}(n, m, nlb, nub, ind_cons)

    obj_scale = T[1.0]
    con_scale = ones(T,m)
    con_jac_scale = ones(T,n_jac)
    @trace(logger,"Initializing linear solver.")
    cnt.linear_solver_time =
        @elapsed linear_solver = opt.linear_solver(get_kkt(kkt) ; opt=opt_linear_solver, logger=logger)

    n_kkt = size(kkt, 1)
    buffer_vec = similar(full(d), n_kkt)
    @trace(logger,"Initializing iterative solver.")
    iterator = opt.iterator(linear_solver, kkt, buffer_vec)

    @trace(logger,"Initializing fixed variable treatment scheme.")

    if opt.inertia_correction_method == INERTIA_AUTO
        opt.inertia_correction_method = is_inertia(linear_solver)::Bool ? INERTIA_BASED : INERTIA_FREE
    end


    return MadNLPSolver{T,KKTSystem,typeof(nlp),typeof(linear_solver),typeof(iterator),typeof(_w1)}(
        nlp,kkt,opt,cnt,logger,
        n,m,nlb,nub,x,y,zl,zu,xl,xu,0.,f,c,
        jacl,
        d, p,
        _w1, _w2, _w3, _w4,
        x_trial,c_trial,0.,c_slk,rhs,
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

