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

@kwdef mutable struct Ma97Control{T}
    f_arrays::Cint = 0
    action::Cint = 0
    nemin::Cint = 0
    multiplier::T = 0.
    ordering::Cint = 0
    print_level::Cint = 0
    scaling::Cint = 0
    small::T = 0
    u::T = 0
    unit_diagnostics::Cint = 0
    unit_error::Cint = 0
    unit_warning::Cint = 0
    factor_min::Clong = 0
    solve_blas3::Cint = 0
    solve_min::Clong = 0
    solve_mf::Cint = 0
    consist_tol::T = 0
    ispare::Vector{Cint}
    rspare::Vector{T}
end

@kwdef mutable struct Ma97Info{T}
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
    rspare::Vector{T}
end

mutable struct Ma97Solver{T} <:AbstractLinearSolver{T}
    n::Int32

    csc::SparseMatrixCSC{T,Int32}

    control::Ma97Control{T}
    info::Ma97Info{T}

    akeep::Vector{Ptr{Nothing}}
    fkeep::Vector{Ptr{Nothing}}

    opt::Ma97Options
    logger::MadNLPLogger
end

for (fdefault, fanalyse, ffactor, fsolve, ffinalise, typ) in [
    (:ma97_default_control_d, :ma97_analyse_d,
     :ma97_factor_d, :ma97_solve_d, :ma97_finalise_d, Float64),
    (:ma97_default_control_s, :ma97_analyse_s,
     :ma97_factor_s, :ma97_solve_s, :ma97_finalise_s, Float32)
     ]
    @eval begin
        ma97_default_control(
            control::Ma97Control{$typ}
        ) = ccall(
            ($(string(fdefault)), libhsl),
            Nothing,
            (Ref{Ma97Control{$typ}},),
            control
        )
        ma97_analyse(
            check::Cint,n::Cint,ptr::Vector{Cint},row::Vector{Cint},
            val::Ptr{Nothing},akeep::Vector{Ptr{Nothing}},
            control::Ma97Control{$typ},info::Ma97Info{$typ},
            order::Ptr{Nothing}
        ) = ccall(
            ($(string(fanalyse)),libhsl),
            Nothing,
            (Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{$typ},
             Ptr{Ptr{Nothing}},Ref{Ma97Control{$typ}},Ref{Ma97Info{$typ}},Ptr{Cint}),
            check,n,ptr,row,val,akeep,control,info,order
        )
        ma97_factor(
            matrix_type::Cint,ptr::Ptr{Nothing},row::Ptr{Nothing},
            val::Vector{$typ},akeep::Vector{Ptr{Nothing}},fkeep::Vector{Ptr{Nothing}},
            control::Ma97Control,info::Ma97Info,scale::Ptr{Nothing}
        ) = ccall(
            ($(string(ffactor)),libhsl),
            Nothing,
            (Cint,Ptr{Cint},Ptr{Cint},Ptr{$typ},Ptr{Ptr{Nothing}},
             Ptr{Ptr{Nothing}},Ref{Ma97Control},Ref{Ma97Info},Ptr{$typ}),
            matrix_type,ptr,row,val,akeep,fkeep,control,info,scale
        )
        ma97_solve(
            job::Cint,nrhs::Cint,x::Vector{$typ},ldx::Cint,
            akeep::Vector{Ptr{Nothing}},fkeep::Vector{Ptr{Nothing}},
            control::Ma97Control,info::Ma97Info
        ) = ccall(
            ($(string(fsolve)),libhsl),
            Nothing,
            (Cint,Cint,Ptr{$typ},Cint,Ptr{Ptr{Nothing}},
             Ptr{Ptr{Nothing}},Ref{Ma97Control},Ref{Ma97Info}),
            job,nrhs,x,ldx,akeep,fkeep,control,info
        )
        ma97_finalize(
            ::Type{$typ},akeep::Vector{Ptr{Nothing}},fkeep::Vector{Ptr{Nothing}}
        )=ccall(
            ($(string(ffinalise)),libhsl),
            Nothing,
            (Ptr{Ptr{Nothing}},Ptr{Ptr{Nothing}}),
            akeep,fkeep
        )
    end
end
ma97_set_num_threads(n) = ccall((:omp_set_num_threads_,libhsl),
                                Cvoid,
                                (Ref{Cint},),
                                Cint(n))


function Ma97Solver(
    csc::SparseMatrixCSC{T,Int32};
    opt=Ma97Options(),logger=MadNLPLogger(),
) where T

    ma97_set_num_threads(opt.ma97_num_threads)

    n = Int32(csc.n)

    info = Ma97Info{T}(ispare=zeros(Int32,5),rspare=zeros(T,10))
    control=Ma97Control{T}(ispare=zeros(Int32,5),rspare=zeros(T,10))
    ma97_default_control(control)

    control.print_level = opt.ma97_print_level
    control.f_arrays = 1
    control.nemin = opt.ma97_nemin
    control.ordering = Int32(opt.ma97_order)
    control.small = opt.ma97_small
    control.u = opt.ma97_u
    control.scaling = Int32(opt.ma97_scaling)

    akeep = [C_NULL]
    fkeep = [C_NULL]

    ma97_analyse(Int32(1),n,csc.colptr,csc.rowval,C_NULL,akeep,control,info,C_NULL)
    info.flag<0 && throw(SymbolicException())
    M = Ma97Solver{T}(n,csc,control,info,akeep,fkeep,opt,logger)
    finalizer(finalize,M)
    return M
end
function factorize!(M::Ma97Solver)
    ma97_factor(Int32(4),C_NULL,C_NULL,M.csc.nzval,M.akeep,M.fkeep,M.control,M.info,C_NULL)
    M.info.flag<0 && throw(FactorizationException())
    return M
end
function solve!(M::Ma97Solver{T},rhs::Vector{T}) where T
    ma97_solve(Int32(0),Int32(1),rhs,M.n,M.akeep,M.fkeep,M.control,M.info)
    M.info.flag<0 && throw(SolveException())
    return rhs
end
is_inertia(::Ma97Solver)=true
function inertia(M::Ma97Solver)
    return (M.info.matrix_rank-M.info.num_neg,M.n-M.info.matrix_rank,M.info.num_neg)
end
finalize(M::Ma97Solver{T}) where T = ma97_finalize(T,M.akeep,M.fkeep)

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
default_options(::Type{Ma97Solver}) = Ma97Options()
is_supported(::Type{Ma97Solver},::Type{Float32}) = true
is_supported(::Type{Ma97Solver},::Type{Float64}) = true
