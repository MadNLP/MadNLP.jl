# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

abstract type AbstractMadNLPSolver{T} end

include("restoration.jl")
include("inertiacorrector.jl")

mutable struct MadNLPSolver{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    KKTSystem <: AbstractKKTSystem{T},
    Model <: AbstractNLPModel{T,VT},
    CB <: AbstractCallback{T},
    Iterator <: AbstractIterator{T},
    IC <: AbstractInertiaCorrector,
    KKTVec <: AbstractKKTVector{T, VT}
    } <: AbstractMadNLPSolver{T}
    
    nlp::Model
    cb::CB
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

    del_w::T
    del_w_last::T
    del_c::T

    filter::Vector{Tuple{T,T}}

    inertia_corrector::IC
    RR::Union{Nothing,RobustRestorer{T}}
    status::Status
    output::Dict
end

function MadNLPSolver(nlp::AbstractNLPModel{T,VT}; kwargs...) where {T, VT}
    
    opt, opt_linear_solver, logger = load_options(nlp; kwargs...)
    @assert is_supported(opt.linear_solver, T)

    cnt = MadNLPCounters(start_time=time())
    cb = create_callback(opt.callback, nlp, opt)
    
    # generic options
    opt.disable_garbage_collector &&
        (GC.enable(false); @warn(logger,"Julia garbage collector is temporarily disabled"))
    set_blas_num_threads(opt.blas_num_threads; permanent=true)
    @trace(logger,"Initializing variables.")
    
    ind_cons = get_index_constraints(
        get_lvar(nlp), get_uvar(nlp),
        get_lcon(nlp), get_ucon(nlp),
        opt.fixed_variable_treatment,
        opt.equality_treatment
    )

    ind_lb = ind_cons.ind_lb
    ind_ub = ind_cons.ind_ub
    
    ns = length(ind_cons.ind_ineq)
    nx = get_nvar(nlp)
    n = nx+ns
    m = get_ncon(nlp)
    nlb = length(ind_lb)
    nub = length(ind_ub)

    @trace(logger,"Initializing KKT system.")
    kkt = create_kkt_system(
        opt.kkt_system,
        cb,
        opt,
        opt_linear_solver,
        cnt,
        ind_cons
    )

    @trace(logger,"Initializing iterative solver.")
    iterator = opt.iterator(kkt; cnt = cnt, logger = logger)

    x = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    xl = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    xu = PrimalVector(VT, nx, ns, ind_lb, ind_ub)  
    zl = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    zu = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    f = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    x_trial = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    
    d = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    p = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w1 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w2 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w3 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w4 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)

    jacl = VT(undef,n) 
    c_trial = VT(undef, m)
    y = VT(undef, m)
    c = VT(undef, m)
    rhs = VT(undef, m)

    c_slk = view(c,ind_cons.ind_ineq)
    x_lr = view(full(x), ind_cons.ind_lb)
    x_ur = view(full(x), ind_cons.ind_ub)
    xl_r = view(full(xl), ind_cons.ind_lb)
    xu_r = view(full(xu), ind_cons.ind_ub)
    zl_r = view(full(zl), ind_cons.ind_lb)
    zu_r = view(full(zu), ind_cons.ind_ub)
    x_trial_lr = view(full(x_trial), ind_cons.ind_lb)
    x_trial_ur = view(full(x_trial), ind_cons.ind_ub)
    dx_lr = view(d.xp, ind_cons.ind_lb) # TODO
    dx_ur = view(d.xp, ind_cons.ind_ub) # TODO

    inertia_correction_method = if opt.inertia_correction_method == InertiaAuto
        is_inertia(kkt.linear_solver)::Bool ? InertiaBased : InertiaFree
    else
        opt.inertia_correction_method
    end

    inertia_corrector = build_inertia_corrector(
        inertia_correction_method,
        VT,
        n, m, nlb, nub, ind_lb, ind_ub
    )
    
    cnt.init_time = time() - cnt.start_time

    return MadNLPSolver(
        nlp, cb, kkt,
        opt, cnt, logger, 
        n, m, nlb, nub,
        x, y, zl, zu, xl, xu,
        zero(T), f, c, 
        jacl, 
        d, p, 
        _w1, _w2, _w3, _w4, 
        x_trial, c_trial, zero(T), c_slk, rhs, 
        ind_cons.ind_ineq, ind_cons.ind_fixed, ind_cons.ind_llb, ind_cons.ind_uub, 
        x_lr, x_ur, xl_r, xu_r, zl_r, zu_r, dx_lr, dx_ur, x_trial_lr, x_trial_ur, 
        iterator, 
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
        " ",
        zero(T), zero(T), zero(T),
        Tuple{T, T}[],
        inertia_corrector, nothing,
        INITIAL, Dict(), 
    )

end

include("utils.jl")
include("kernels.jl")
include("callbacks.jl")
include("factorization.jl")
include("solver.jl")


