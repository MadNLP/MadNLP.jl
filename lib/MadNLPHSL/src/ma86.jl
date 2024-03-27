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

mutable struct Ma86Solver{T} <: AbstractLinearSolver{T}
    csc::SparseMatrixCSC{T,Int32}

    control::ma86_control{T}
    info::ma86_info{T}

    mc68_control::mc68_control
    mc68_info::mc68_info

    order::Vector{Int32}
    keep::Vector{Ptr{Nothing}}

    opt::Ma86Options
    logger::MadNLPLogger
end


for (fdefault, fanalyse, ffactor, fsolve, ffinalise, typ) in [
    (:ma86_default_control_d, :ma86_analyse_d,
     :ma86_factor_d, :ma86_solve_d, :ma86_finalise_d, Float64),
    (:ma86_default_control_s, :ma86_analyse_s,
     :ma86_factor_s, :ma86_solve_s, :ma86_finalise_s, Float32)
     ]
    @eval begin
        ma86_default_control(
            control::ma86_control{$typ}
        ) = HSL.$fdefault(control)

        ma86_analyse(
            n::Cint,colptr::Vector{Cint},rowval::Vector{Cint},
            order::Vector{Cint},keep::Vector{Ptr{Nothing}},
            control::ma86_control{$typ},info::ma86_info{$typ}
        ) = HSL.$fanalyse(n,colptr,rowval,order,keep,control,info)

        ma86_factor(
            n::Cint,colptr::Vector{Cint},rowval::Vector{Cint},
            nzval::Vector{$typ},order::Vector{Cint},
            keep::Vector{Ptr{Nothing}},control::ma86_control,info::ma86_info,
            scale::Ptr{Nothing}
        ) = HSL.$ffactor(n,colptr,rowval,nzval,order,keep,control,info,scale)

        ma86_solve(
            job::Cint,nrhs::Cint,n::Cint,rhs::Vector{$typ},
            order::Vector{Cint},keep::Vector{Ptr{Nothing}},
            control::ma86_control,info::ma86_info,scale::Ptr{Nothing}
        ) = HSL.$fsolve(job,nrhs,n,rhs,order,keep,control,info,scale)

        ma86_finalize(
            keep::Vector{Ptr{Nothing}},control::ma86_control{$typ}
        ) = HSL.$ffinalise(keep,control)
    end
end
ma86_set_num_threads(n) = HSL.omp_set_num_threads(n)

function Ma86Solver(
    csc::SparseMatrixCSC{T,Int32};
    opt=Ma86Options(),logger=MadNLPLogger(),
) where T

    ma86_set_num_threads(opt.ma86_num_threads)

    order = Vector{Int32}(undef,csc.n)

    info = ma86_info{T}()
    control = ma86_control{T}()
    mc68info = mc68_info()
    mc68control = mc68_control()
    HSL.mc68_default_control_i(mc68control)

    keep = [C_NULL]

    mc68control.f_array_in=1
    mc68control.f_array_out=1
    HSL.mc68_order_i(Int32(opt.ma86_order),Int32(csc.n),csc.colptr,csc.rowval,order,mc68control,mc68info)

    ma86_default_control(control)
    control.diagnostics_level = Int32(opt.ma86_print_level)
    control.f_arrays = 1
    control.nemin = opt.ma86_nemin
    control.small_ = opt.ma86_small
    control.u = opt.ma86_u
    control.scaling = Int32(opt.ma86_scaling)

    ma86_analyse(Int32(csc.n),csc.colptr,csc.rowval,order,keep,control,info)
    info.flag<0 && throw(SymbolicException())

    M = Ma86Solver{T}(csc,control,info,mc68control,mc68info,order,keep,opt,logger)
    finalizer(finalize,M)

    return M
end
function factorize!(M::Ma86Solver)
    ma86_factor(Int32(M.csc.n),M.csc.colptr,M.csc.rowval,M.csc.nzval,
                  M.order,M.keep,M.control,M.info,C_NULL)
    M.info.flag<0 && throw(FactorizationException())
    return M
end
function solve!(M::Ma86Solver{T},rhs::Vector{T}) where T
    ma86_solve(Int32(0),Int32(1),Int32(M.csc.n),rhs,
               M.order,M.keep,M.control,M.info,C_NULL)
    M.info.flag<0 && throw(SolveException())
    return rhs
end
is_inertia(::Ma86Solver)=true
function inertia(M::Ma86Solver)
    return (M.info.matrix_rank-M.info.num_neg,M.csc.n-M.info.matrix_rank,M.info.num_neg)
end
finalize(M::Ma86Solver) = ma86_finalize(M.keep,M.control)
function improve!(M::Ma86Solver)
    if M.control.u == M.opt.ma86_umax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.control.u = min(M.opt.ma86_umax,M.control.u^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.control.u)")
    return true
end
introduce(::Ma86Solver)="ma86 v$(HSL.HSL_MA86_version())"
input_type(::Type{Ma86Solver}) = :csc
default_options(::Type{Ma86Solver}) = Ma86Options()
is_supported(::Type{Ma86Solver},::Type{Float32}) = true
is_supported(::Type{Ma86Solver},::Type{Float64}) = true

