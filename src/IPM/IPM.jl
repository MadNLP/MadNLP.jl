# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

abstract type AbstractMadNLPSolver{T} end

include("restoration.jl")

mutable struct MadNLPSolver{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    KKTSystem <: AbstractKKTSystem{T},
    Model <: AbstractNLPModel,
    Iterator <: AbstractIterator{T},
    KKTVec <: AbstractKKTVector{T, VT}
    } <: AbstractMadNLPSolver{T}
    
    nlp::Model
    kkt::KKTSystem

    opt::MadNLPOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger

    n::Int # number of variables (after reformulation)
    m::Int # number of cons
    nlb::Int
    nub::Int

    x::PrimalVector{T, VT, VI} # primal (after reformulation)
    y::VT # dual
    zl::PrimalVector{T, VT, VI} # dual (after reformulation)
    zu::PrimalVector{T, VT, VI} # dual (after reformulation)
    xl::PrimalVector{T, VT, VI} # primal lower bound (after reformulation)
    xu::PrimalVector{T, VT, VI} # primal upper bound (after reformulation)

    obj_val::T
    f::PrimalVector{T, VT, VI}
    c::VT

    jacl::VT

    d::KKTVec
    p::KKTVec

    _w1::KKTVec
    _w2::KKTVec

    _w3::KKTVec
    _w4::KKTVec

    x_trial::PrimalVector{T, VT, VI}
    c_trial::VT
    obj_val_trial::T

    c_slk::SubVector{T,VT,VI}
    rhs::VT

    ind_ineq::VI
    ind_fixed::VI
    ind_llb::VI
    ind_uub::VI

    x_lr::SubVector{T,VT,VI}
    x_ur::SubVector{T,VT,VI}
    xl_r::SubVector{T,VT,VI}
    xu_r::SubVector{T,VT,VI}
    zl_r::SubVector{T,VT,VI}
    zu_r::SubVector{T,VT,VI}

    dx_lr::SubVector{T,VT,VI}
    dx_ur::SubVector{T,VT,VI}
    x_trial_lr::SubVector{T,VT,VI}
    x_trial_ur::SubVector{T,VT,VI}

    # linear_solver::LinSolver
    iterator::Iterator

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

    # del_w::T
    # del_c::T
    # del_w_last::T

    filter::Vector{Tuple{T,T}}

    RR::Union{Nothing,RobustRestorer{T}}
    status::Status
    output::Dict
end

function MadNLPSolver(nlp::AbstractNLPModel{T,VT}; kwargs...) where {T, VT}
    
    opt, opt_linear_solver, logger = load_options(; kwargs...)
    @assert is_supported(opt.linear_solver, T)

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
    kkt = create_kkt_system(opt.kkt_system, nlp, opt, cnt, ind_cons)

    # Primal variable
    x = PrimalVector(VT,nx, ns, ind_cons)
    variable(x) .= get_x0(nlp)
    # Bounds
    xl = PrimalVector(VT,nx, ns, ind_cons)
    variable(xl) .= get_lvar(nlp)
    slack(xl) .= view(get_lcon(nlp), ind_cons.ind_ineq)
    xu = PrimalVector(VT,nx, ns, ind_cons)
    variable(xu) .= get_uvar(nlp)
    slack(xu) .= view(get_ucon(nlp), ind_cons.ind_ineq)
    zl = PrimalVector(VT,nx, ns, ind_cons)
    zu = PrimalVector(VT,nx, ns, ind_cons)
    
    # Gradient
    f = PrimalVector(VT,nx, ns, ind_cons)

    y = copy(get_y0(nlp))
    c = VT(undef, m)

    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    x_trial = PrimalVector(VT,nx, ns, ind_cons)
    c_trial = VT(undef, m)

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

    _w1 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_cons)
    _w2 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_cons)
    _w3 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_cons)
    _w4 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_cons)

    jacl = fill!(VT(undef,n), zero(T)) # spblas may throw an error if not initialized to zero

    d = UnreducedKKTVector(VT, n, m, nlb, nub, ind_cons)
    dx_lr = view(d.xp, ind_cons.ind_lb) # TODO
    dx_ur = view(d.xp, ind_cons.ind_ub) # TODO

    p = UnreducedKKTVector(VT, n, m, nlb, nub, ind_cons)

    n_kkt = size(kkt, 1)
    @trace(logger,"Initializing iterative solver.")
    residual = UnreducedKKTVector(VT, n, m, nlb, nub, ind_cons)
    iterator = opt.iterator(kkt, residual; cnt = cnt, logger = logger)


    @trace(logger,"Initializing fixed variable treatment scheme.")

    if opt.inertia_correction_method == INERTIA_AUTO
        opt.inertia_correction_method = is_inertia(kkt.linear_solver)::Bool ? INERTIA_BASED : INERTIA_FREE
    end


    return MadNLPSolver(
        nlp,kkt,opt,cnt,logger,
        n,m,nlb,nub,x,y,zl,zu,xl,xu,0.,f,c,
        jacl,
        d, p,
        _w1, _w2, _w3, _w4,
        x_trial,c_trial,0.,c_slk,rhs,
        ind_cons.ind_ineq,ind_cons.ind_fixed,ind_cons.ind_llb,ind_cons.ind_uub,
        x_lr,x_ur,xl_r,xu_r,zl_r,zu_r,dx_lr,dx_ur,x_trial_lr,x_trial_ur,
        iterator,
        0.,0.,0.,0.,0.,0.,0.,0.,0.," ",
        Tuple{T,T}[],nothing,INITIAL,Dict(),
    )

end

include("utils.jl")
include("kernels.jl")
include("callbacks.jl")
include("factorization.jl")
include("solver.jl")


