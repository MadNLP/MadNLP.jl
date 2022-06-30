@kwdef mutable struct Ma97Options <: AbstractOptions
    ma97_num_threads::Int = 1
    ma97_print_level::Int = -1
    ma97_nemin::Int = 8
    ma97_small::Float64 = 1e-20
    ma97_order::Ordering = METIS
    ma97_scaling::Scaling = SCALING_NONE
    ma97_u::Float64 = 1e-8
    ma97_umax::Float64 = 1e-4
end

@kwdef mutable struct Ma97Control
    f_arrays::Cint = 0
    action::Cint = 0
    nemin::Cint = 0
    multiplier::Cdouble = 0.
    ordering::Cint = 0
    print_level::Cint = 0
    scaling::Cint = 0
    small::Cdouble = 0
    u::Cdouble = 0
    unit_diagnostics::Cint = 0
    unit_error::Cint = 0
    unit_warning::Cint = 0
    factor_min::Clong = 0
    solve_blas3::Cint = 0
    solve_min::Clong = 0
    solve_mf::Cint = 0
    consist_tol::Cdouble = 0
    ispare::Vector{Cint}
    rspare::Vector{Cdouble}
end

@kwdef mutable struct Ma97Info
    flag::Cint = 0
    flag68::Cint = 0
    flag77::Cint = 0
    matrix_dup::Cint = 0
    matrix_rank::Cint = 0
    matrix_outrange::Cint = 0
    matrix_missing_diag::Cint = 0
    maxdepth::Cint = 0
    maxfront::Cint = 0
    num_delay::Cint = 0
    num_factor::Clong = 0
    num_flops::Clong = 0
    num_neg::Cint = 0
    num_sup::Cint = 0
    num_two::Cint = 0
    ordering::Cint = 0
    stat::Cint = 0
    ispare::Vector{Cint}
    rspare::Vector{Cdouble}
end

mutable struct Ma97Solver <:AbstractLinearSolver
    n::Int32

    csc::SparseMatrixCSC{Float64,Int32}

    control::Ma97Control
    info::Ma97Info

    akeep::Vector{Ptr{Nothing}}
    fkeep::Vector{Ptr{Nothing}}

    opt::Ma97Options
    logger::Logger
end

ma97_default_control_d(control::Ma97Control) = ccall(
    (:ma97_default_control_d,libhsl),
    Nothing,
    (Ref{Ma97Control},),
    control)

ma97_analyse_d(check::Cint,n::Cint,ptr::Vector{Cint},row::Vector{Cint},
               val::Ptr{Nothing},akeep::Vector{Ptr{Nothing}},
               control::Ma97Control,info::Ma97Info,
               order::Ptr{Nothing}) = ccall(
                         (:ma97_analyse_d,libhsl),
                         Nothing,
                         (Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cdouble},
                          Ptr{Ptr{Nothing}},Ref{Ma97Control},Ref{Ma97Info},Ptr{Cint}),
                         check,n,ptr,row,val,akeep,control,info,order)
ma97_factor_d(matrix_type::Cint,ptr::Ptr{Nothing},row::Ptr{Nothing},
              val::Vector{Cdouble},akeep::Vector{Ptr{Nothing}},fkeep::Vector{Ptr{Nothing}},
              control::Ma97Control,info::Info,scale::Ptr{Nothing}) = ccall(
                  (:ma97_factor_d,libhsl),
                  Nothing,
                  (Cint,Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Ptr{Nothing}},
                   Ptr{Ptr{Nothing}},Ref{Ma97Control},Ref{Ma97Info},Ptr{Cdouble}),
                  matrix_type,ptr,row,val,akeep,fkeep,control,info,scale)
ma97_solve_d(job::Cint,nrhs::Cint,x::Vector{Cdouble},ldx::Cint,
             akeep::Vector{Ptr{Nothing}},fkeep::Vector{Ptr{Nothing}},
             control::Ma97Control,info::Ma97Info) = ccall(
                 (:ma97_solve_d,libhsl),
                 Nothing,
                 (Cint,Cint,Ptr{Cdouble},Cint,Ptr{Ptr{Nothing}},
                  Ptr{Ptr{Nothing}},Ref{Ma97Control},Ref{Ma97Info}),
                 job,nrhs,x,ldx,akeep,fkeep,control,info)
ma97_finalize_d(akeep::Vector{Ptr{Nothing}},fkeep::Vector{Ptr{Nothing}})=ccall(
    (:ma97_finalise_d,libhsl),
    Nothing,
    (Ptr{Ptr{Nothing}},Ptr{Ptr{Nothing}}),
    akeep,fkeep)
ma97_set_num_threads(n) = ccall((:omp_set_num_threads_,libhsl),
                                Cvoid,
                                (Ref{Cint},),
                                Cint(n))


function Ma97Solver(csc::SparseMatrixCSC{Float64,Int32};
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Ma97Options(),logger=Logger(),
                kwargs...)

    set_options!(opt,option_dict,kwargs)

    ma97_set_num_threads(opt.ma97_num_threads)

    n = Int32(csc.n)

    info = Ma97Info(ispare=zeros(Int32,5),rspare=zeros(Float64,10))
    control=Ma97Control(ispare=zeros(Int32,5),rspare=zeros(Float64,10))
    ma97_default_control_d(control)

    control.print_level = opt.ma97_print_level
    control.f_arrays = 1
    control.nemin = opt.ma97_nemin
    control.ordering = Int32(opt.ma97_order)
    control.small = opt.ma97_small
    control.u = opt.ma97_u
    control.scaling = Int32(opt.ma97_scaling)

    akeep = [C_NULL]
    fkeep = [C_NULL]

    ma97_analyse_d(Int32(1),n,csc.colptr,csc.rowval,C_NULL,akeep,control,info,C_NULL)
    info.flag<0 && throw(SymbolicException())
    M = Ma97Solver(n,csc,control,info,akeep,fkeep,opt,logger)
    finalizer(finalize,M)
    return M
end
function factorize!(M::Ma97Solver)
    ma97_factor_d(Int32(4),C_NULL,C_NULL,M.csc.nzval,M.akeep,M.fkeep,M.control,M.info,C_NULL)
    M.info.flag<0 && throw(FactorizationException())
    return M
end
function solve!(M::Ma97Solver,rhs::Vector{Float64})
    ma97_solve_d(Int32(0),Int32(1),rhs,M.n,M.akeep,M.fkeep,M.control,M.info)
    M.info.flag<0 && throw(SolveException())
    return rhs
end
is_inertia(::Ma97Solver)=true
function inertia(M::Ma97Solver)
    return (M.info.matrix_rank-M.info.num_neg,M.n-M.info.matrix_rank,M.info.num_neg)
end
finalize(M::Ma97Solver) = ma97_finalize_d(M.akeep,M.fkeep)

function improve!(M::Ma97Solver)
    if M.control.u == M.opt.ma97_umax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.control.u = min(M.opt.ma97_umax,M.control.u^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.control.u)")
    return true
end
introduce(::Ma97Solver)="ma97"
input_type(::Type{Ma97Solver}) = :csc

