# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

@kwdef mutable struct Ma86Options <: AbstractOptions
    ma86_num_threads::Int = 1
    ma86_print_level::Float64 = -1
    ma86_nemin::Int = 32
    ma86_order::Ordering = METIS
    ma86_scaling::Scaling = SCALING_NONE
    ma86_small::Float64 = 1e-20
    ma86_static::Float64 = 0.
    ma86_u::Float64 = 1e-8
    ma86_umax::Float64 = 1e-4
end

@kwdef mutable struct Ma86Control
    f_arrays::Int32 = 0
    diagnostics_level::Int32 = 0
    unit_diagnostics::Int32 = 0
    unit_error::Int32 = 0
    unit_warning::Int32 = 0
    nemin::Int32 = 0
    nb::Int32 = 0
    action::Int32 = 0
    nbi::Int32 = 0
    pool_size::Int32 = 0
    small::Float64 = 0.
    static::Float64 = 0.
    u::Float64 = 0.
    umin::Float64 = 0.
    scaling::Int32 = 0
end

@kwdef mutable struct Ma86Info
    detlog::Float64 = 0.
    detsign::Int32 = 0
    flag::Int32 = 0
    matrix_rank::Int32 = 0
    maxdepth::Int32 = 0
    num_delay::Int32 = 0
    num_factor::Clong = 0
    num_flops::Clong = 0
    num_neg::Int32 = 0
    num_nodes::Int32 = 0
    num_nothresh::Int32 = 0
    num_perturbed::Int32 = 0
    num_two::Int32 = 0
    pool_size::Int32 = 0
    stat::Int32 = 0
    usmall::Float64 = 0.
end

mutable struct Ma86Solver<:AbstractLinearSolver
    csc::SparseMatrixCSC{Float64,Int32}

    control::Ma86Control
    info::Ma86Info

    mc68_control::Mc68Control
    mc68_info::Mc68Info

    order::Vector{Int32}
    keep::Vector{Ptr{Nothing}}

    opt::Ma86Options
    logger::Logger
end


ma86_default_control_d(control::Ma86Control) = ccall(
    (:ma86_default_control_d,libhsl),
    Nothing,
    (Ref{Ma86Control},),
    control)
ma86_analyse_d(n::Cint,colptr::Vector{Cint},rowval::Vector{Cint},
               order::Vector{Cint},keep::Vector{Ptr{Nothing}},
               control::Ma86Control,info::Info) = ccall(
                   (:ma86_analyse_d,libhsl),
                   Nothing,
                   (Cint,Ptr{Cint},Ptr{Cint},Ptr{Cdouble},
                    Ptr{Ptr{Nothing}},Ref{Ma86Control},Ref{Ma86Info}),
                   n,colptr,rowval,order,keep,control,info)
ma86_factor_d(n::Cint,colptr::Vector{Cint},rowval::Vector{Cint},
              nzval::Vector{Cdouble},order::Vector{Cint},
              keep::Vector{Ptr{Nothing}},control::Ma86Control,info::Ma86Info,
              scale::Ptr{Nothing}) = ccall(
                  (:ma86_factor_d,libhsl),
                  Nothing,
                  (Cint,Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},
                   Ptr{Ptr{Nothing}},Ref{Ma86Control},Ref{Ma86Info},Ptr{Nothing}),
                  n,colptr,rowval,nzval,order,keep,control,info,scale)
ma86_solve_d(job::Cint,nrhs::Cint,n::Cint,rhs::Vector{Cdouble},
             order::Vector{Cint},keep::Vector{Ptr{Nothing}},
             control::Ma86Control,info::Ma86Info,scale::Ptr{Nothing}) = ccall(
                 (:ma86_solve_d,libhsl),
                 Nothing,
                 (Cint,Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Ptr{Nothing}},
                  Ref{Ma86Control},Ref{Ma86Info},Ptr{Nothing}),
                 job,nrhs,n,rhs,order,keep,control,info,scale)
ma86_finalize_d(keep::Vector{Ptr{Nothing}},control::Ma86Control)=ccall(
    (:ma86_finalise_d,libhsl),
    Nothing,
    (Ptr{Ptr{Nothing}},Ref{Ma86Control}),
    keep,control)

ma86_set_num_threads(n) = ccall((:omp_set_num_threads_,libhsl),
                                Cvoid,
                                (Ref{Int32},),
                                Int32(n))

function Ma86Solver(csc::SparseMatrixCSC{Float64,Int32};
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Ma86Options(),logger=Logger(),
                kwargs...)

    set_options!(opt,option_dict,kwargs)

    ma86_set_num_threads(opt.ma86_num_threads)

    order = Vector{Int32}(undef,csc.n)

    info=Ma86Info()
    control=Ma86Control()
    mc68_info = Mc68Info()
    mc68_control = get_mc68_default_control()

    keep = [C_NULL]

    mc68_control.f_array_in=1
    mc68_control.f_array_out=1
    mc68_order_i(Int32(opt.ma86_order),Int32(csc.n),csc.colptr,csc.rowval,order,mc68_control,mc68_info)

    ma86_default_control_d(control)
    control.diagnostics_level = Int32(opt.ma86_print_level)
    control.f_arrays = 1
    control.nemin = opt.ma86_nemin
    control.small = opt.ma86_small
    control.u = opt.ma86_u
    control.scaling = Int32(opt.ma86_scaling)

    ma86_analyse_d(Int32(csc.n),csc.colptr,csc.rowval,order,keep,control,info)
    info.flag<0 && throw(SymbolicException())

    M = Ma86Solver(csc,control,info,mc68_control,mc68_info,order,keep,opt,logger)
    finalizer(finalize,M)

    return M
end
function factorize!(M::Ma86Solver)
    ma86_factor_d(Int32(M.csc.n),M.csc.colptr,M.csc.rowval,M.csc.nzval,
                  M.order,M.keep,M.control,M.info,C_NULL)
    M.info.flag<0 && throw(FactorizationException())
    return M
end
function solve!(M::Ma86Solver,rhs::Vector{Float64})
    ma86_solve_d(Int32(0),Int32(1),Int32(M.csc.n),rhs,
                 M.order,M.keep,M.control,M.info,C_NULL)
    M.info.flag<0 && throw(SolveException())
    return rhs
end
is_inertia(::Ma86Solver)=true
function inertia(M::Ma86Solver)
    return (M.info.matrix_rank-M.info.num_neg,M.csc.n-M.info.matrix_rank,M.info.num_neg)
end
finalize(M::Ma86Solver) = ma86_finalize_d(M.keep,M.control)
function improve!(M::Ma86Solver)
    if M.control.u == M.opt.ma86_umax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.control.u = min(M.opt.ma86_umax,M.control.u^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.control.u)")
    return true
end
introduce(::Ma86Solver)="ma86"
input_type(::Type{Ma86Solver}) = :csc

