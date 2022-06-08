# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module MadNLPMa77

import ..MadNLPHSL:
    @kwdef, Logger, @debug, @warn, @error, libhsl,
    SparseMatrixCSC, SparseMatrixCSC, SubVector,
    get_tril_to_full, transfer!,
    AbstractOptions, AbstractLinearSolver, set_options!,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia
import ..MadNLPHSL: Mc68

const INPUT_MATRIX_TYPE = :csc

@enum(Ordering::Int,AMD = 1, METIS = 3)

@kwdef mutable struct Options <: AbstractOptions
    ma77_buffer_lpage::Int = 4096
    ma77_buffer_npage::Int = 1600
    ma77_file_size::Int = 2097152
    ma77_maxstore::Int = 0
    ma77_nemin::Int = 8
    ma77_order::Ordering = METIS
    ma77_print_level::Int = -1
    ma77_small::Float64 = 1e-20
    ma77_static::Float64 = 0.
    ma77_u::Float64 = 1e-8
    ma77_umax::Float64 = 1e-4
end

@kwdef mutable struct Control
    f_arrays::Cint = 0
    print_level::Cint = 0
    unit_diagnostics::Cint = 0
    unit_error::Cint = 0
    unit_warning::Cint = 0
    bits::Cint = 0

    buffer_lpage_1::Cint = 0
    buffer_lpage_2::Cint = 0
    buffer_npage_1::Cint = 0
    buffer_npage_2::Cint = 0

    file_size::Clong = 0
    maxstore::Clong = 0

    storage_1::Clong = 0
    storage_2::Clong = 0
    storage_3::Clong = 0

    nemin::Cint = 0
    maxit::Cint = 0
    infnorm::Cint = 0
    thresh::Cdouble = 0.
    nb54::Cint = 0
    action::Cint = 0
    multiplier::Cdouble = 0.
    nb64::Cint = 0
    nbi::Cint = 0
    small::Cdouble = 0.
    static_::Cdouble = 0.
    storage_indef::Clong = 0
    u::Cdouble = 0.
    umin::Cdouble = 0.
    consist_tol::Cdouble = 0.

    ispare_1::Cint = 0
    ispare_2::Cint = 0
    ispare_3::Cint = 0
    ispare_4::Cint = 0
    ispare_5::Cint = 0

    lspare_1::Clong = 0
    lspare_2::Clong = 0
    lspare_3::Clong = 0
    lspare_4::Clong = 0
    lspare_5::Clong = 0

    rspare_1::Cdouble = 0.
    rspare_2::Cdouble = 0.
    rspare_3::Cdouble = 0.
    rspare_4::Cdouble = 0.
    rspare_5::Cdouble = 0.
end

@kwdef mutable struct Info
    detlog::Cdouble = 0.
    detsign::Cint = 0
    flag::Cint = 0
    iostat::Cint = 0
    matrix_dup::Cint = 0
    matrix_rank::Cint = 0
    matrix_outrange::Cint = 0
    maxdepth::Cint = 0
    maxfront::Cint = 0
    minstore::Clong = 0
    ndelay::Cint = 0
    nfactor::Clong = 0
    nflops::Clong = 0
    niter::Cint = 0
    nsup::Cint = 0
    num_neg::Cint = 0
    num_nothresh::Cint = 0
    num_perturbed::Cint = 0
    ntwo::Cint = 0

    stat_1::Cint = 0
    stat_2::Cint = 0
    stat_3::Cint = 0
    stat_4::Cint = 0

    nio_read_1::Clong = 0 # 2
    nio_read_2::Clong = 0 # 2

    nio_write_1::Clong = 0 # 2
    nio_write_2::Clong = 0 # 2

    nwd_read_1::Clong = 0 # 2
    nwd_read_2::Clong = 0 # 2

    nwd_write_1::Clong = 0 # 2
    nwd_write_2::Clong = 0 # 2

    num_file_1::Cint = 0 # 4
    num_file_2::Cint = 0 # 4
    num_file_3::Cint = 0 # 4
    num_file_4::Cint = 0 # 4

    storage_1::Clong = 0 # 4
    storage_2::Clong = 0 # 4
    storage_3::Clong = 0 # 4
    storage_4::Clong = 0 # 4

    tree_nodes::Cint = 0
    unit_restart::Cint = 0
    unused::Cint = 0
    usmall::Cdouble = 0.


    ispare_1::Cint = 0
    ispare_2::Cint = 0
    ispare_3::Cint = 0
    ispare_4::Cint = 0
    ispare_5::Cint = 0

    lspare_1::Clong = 0
    lspare_2::Clong = 0
    lspare_3::Clong = 0
    lspare_4::Clong = 0
    lspare_5::Clong = 0

    rspare_1::Cdouble = 0.
    rspare_2::Cdouble = 0.
    rspare_3::Cdouble = 0.
    rspare_4::Cdouble = 0.
    rspare_5::Cdouble = 0.
end

mutable struct Solver <: AbstractLinearSolver
    tril::SparseMatrixCSC{Float64,Int32}
    full::SparseMatrixCSC{Float64,Int32}
    tril_to_full_view::SubVector{Float64}

    control::Control
    info::Info

    mc68_control::Mc68.Control
    mc68_info::Mc68.Info

    order::Vector{Int32}
    keep::Vector{Ptr{Nothing}}

    opt::Options
    logger::Logger
end

ma77_default_control_d(control::Control) = ccall(
    (:ma77_default_control_d,libhsl),
    Cvoid,
    (Ref{Control},),
    control)
ma77_open_d(n::Cint,fname1::String,fname2::String,fname3::String,fname4::String,
            keep::Vector{Ptr{Cvoid}},control::Control,info::Info) = ccall(
                (:ma77_open_d,libhsl),
                Cvoid,
                (Cint,Ptr{Cchar},Ptr{Cchar},Ptr{Cchar},Ptr{Cchar},
                 Ptr{Ptr{Cvoid}},Ref{Control},Ref{Info}),
                n,fname1,fname2,fname3,fname4,keep,control,info)
ma77_input_vars_d(idx::Cint,nvar::Cint,list::AbstractVector{Cint},
                  keep::Vector{Ptr{Cvoid}},control::Control,info::Info) = ccall(
                      (:ma77_input_vars_d,libhsl),
                      Cvoid,
                      (Cint,Cint,Ptr{Cint},
                       Ptr{Ptr{Cvoid}},Ref{Control},Ref{Info}),
                      idx,nvar,list,keep,control,info)
ma77_input_reals_d(idx::Cint,length::Cint,reals::AbstractVector{Cdouble},
                   keep::Vector{Ptr{Cvoid}},control::Control,info::Info) = ccall(
                       (:ma77_input_reals_d,libhsl),
                       Cvoid,
                       (Cint,Cint,Ptr{Cdouble},
                        Ptr{Ptr{Cvoid}},Ref{Control},Ref{Info}),
                       idx,length,reals,keep,control,info)
ma77_analyse_d(order::Vector{Cint},
               keep::Vector{Ptr{Cvoid}},control::Control,info::Info) = ccall(
                   (:ma77_analyse_d,libhsl),
                   Cvoid,
                   (Ptr{Cint},Ptr{Ptr{Cvoid}},Ref{Control},Ref{Info}),
                   order,keep,control,info)
ma77_factor_d(posdef::Cint,keep::Vector{Ptr{Cvoid}},control::Control,info::Info,
              scale::Ptr{Nothing}) = ccall(
                  (:ma77_factor_d,libhsl),
                  Cvoid,
                  (Cint,Ptr{Ptr{Cvoid}},Ref{Control},Ref{Info},Ptr{Nothing}),
                  posdef,keep,control,info,scale)
ma77_factor_solve_d(posdef::Cint,keep::Vector{Ptr{Cvoid}},control::Control,info::Info,
                    scale::Vector{Cdouble},nrhs::Cint,lx::Cint,rhs::Vector{Cdouble}) = ccall(
                        (:ma77_factor_solve_d,libhsl),
                        Cvoid,
                        (Cint,Ptr{Ptr{Cvoid}},Ref{Control},Ref{Info},
                         Ptr{Cdouble},Cint,Cint,Ptr{Cdouble}),
                        posdef,keep,control,info,scale,nrhs,lx,rhs);
ma77_solve_d(job::Cint,nrhs::Cint,lx::Cint,x::Vector{Cdouble},
             keep::Vector{Ptr{Cvoid}},control::Control,info::Info,
             scale::Ptr{Nothing})=ccall(
                 (:ma77_solve_d,libhsl),
                 Cvoid,
                 (Cint,Cint,Cint,Ptr{Float64},
                  Ptr{Ptr{Cvoid}},Ref{Control},Ref{Info},Ptr{Nothing}),
                 job,nrhs,lx,x,keep,control,info,scale);
ma77_finalize_d(keep::Vector{Ptr{Cvoid}},control::Control,info::Info) = ccall(
    (:ma77_finalise_d,libhsl),
    Cvoid,
    (Ptr{Ptr{Cvoid}},Ref{Control},Ref{Info}),
    keep,control,info)

function Solver(csc::SparseMatrixCSC{Float64,Int32};
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Options(),logger=Logger(),
                kwargs...)

    set_options!(opt,option_dict,kwargs)

    full,tril_to_full_view = get_tril_to_full(csc)
    order = Vector{Int32}(undef,csc.n)

    mc68_info = Mc68.Info()
    mc68_control = Mc68.get_mc68_default_control()

    keep = [C_NULL]

    mc68_control.f_array_in=1
    mc68_control.f_array_out=1
    Mc68.mc68_order_i(Int32(opt.ma77_order),Int32(csc.n),csc.colptr,csc.rowval,order,mc68_control,mc68_info)

    info=Info()
    control=Control()
    ma77_default_control_d(control)
    control.f_arrays = 1
    control.bits = 32
    control.file_size = opt.ma77_file_size
    control.maxstore = opt.ma77_maxstore
    control.print_level = -1

    control.buffer_lpage_1=opt.ma77_buffer_lpage
    control.buffer_lpage_2=opt.ma77_buffer_lpage
    control.buffer_npage_1=opt.ma77_buffer_npage
    control.buffer_npage_2=opt.ma77_buffer_npage

    control.nemin = opt.ma77_nemin
    control.small = opt.ma77_small
    control.static_ = opt.ma77_static
    control.u = opt.ma77_u

    isfile(".ma77_int")   && rm(".ma77_int")
    isfile(".ma77_real")  && rm(".ma77_real")
    isfile(".ma77_work")  && rm(".ma77_work")
    isfile(".ma77_delay") && rm(".ma77_delay")

    ma77_open_d(Int32(full.n),".ma77_int", ".ma77_real", ".ma77_work", ".ma77_delay",
                keep,control,info)

    info.flag < 0 && throw(SymbolicException())

    for i=1:full.n
        ma77_input_vars_d(Int32(i),full.colptr[i+1]-full.colptr[i],
                          view(full.rowval,full.colptr[i]:full.colptr[i+1]-1),
                          keep,control,info);
        info.flag < 0 && throw(LinearSymbolicException())
    end

    ma77_analyse_d(order,keep,control,info)
    info.flag<0 && throw(SymbolicException())

    M = Solver(csc,full,tril_to_full_view,
               control,info,mc68_control,mc68_info,order,keep,opt,logger)
    finalizer(finalize,M)
    return M
end

function factorize!(M::Solver)
    M.full.nzval.=M.tril_to_full_view
    for i=1:M.full.n
        ma77_input_reals_d(Int32(i),M.full.colptr[i+1]-M.full.colptr[i],
                           view(M.full.nzval,M.full.colptr[i]:M.full.colptr[i+1]-1),
                           M.keep,M.control,M.info)
        M.info.flag < 0 && throw(FactorizationException())
    end
    ma77_factor_d(Int32(0),M.keep,M.control,M.info,C_NULL)
    M.info.flag < 0 && throw(FactorizationException())
    return M
end
function solve!(M::Solver,rhs::AbstractVector{Float64})
    ma77_solve_d(Int32(0),Int32(1),Int32(M.full.n),rhs,M.keep,M.control,M.info,C_NULL);
    M.info.flag < 0 && throw(SolveException())
    return rhs
end

is_inertia(::Solver) = true
function inertia(M::Solver)
    return (M.info.matrix_rank-M.info.num_neg,M.full.n-M.info.matrix_rank,M.info.num_neg)
end

finalize(M::Solver) = ma77_finalize_d(M.keep,M.control,M.info)

function improve!(M::Solver)
    if M.control.u == M.opt.ma77_umax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.control.u = min(M.opt.ma77_umax,M.control.u^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.control.u)")
    return true
end

introduce(::Solver)="ma77"

end
