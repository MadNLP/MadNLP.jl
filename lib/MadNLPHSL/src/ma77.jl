@kwdef mutable struct Ma77Options <: AbstractOptions
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

@kwdef mutable struct Ma77Control{T}
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
    thresh::T = 0.
    nb54::Cint = 0
    action::Cint = 0
    multiplier::T = 0.
    nb64::Cint = 0
    nbi::Cint = 0
    small::T = 0.
    static_::T = 0.
    storage_indef::Clong = 0
    u::T = 0.
    umin::T = 0.
    consist_tol::T = 0.

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

    rspare_1::T = 0.
    rspare_2::T = 0.
    rspare_3::T = 0.
    rspare_4::T = 0.
    rspare_5::T = 0.
end

@kwdef mutable struct Ma77Info{T}
    detlog::T = 0.
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
    usmall::T = 0.


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

    rspare_1::T = 0.
    rspare_2::T = 0.
    rspare_3::T = 0.
    rspare_4::T = 0.
    rspare_5::T = 0.
end

mutable struct Ma77Solver{T} <: AbstractLinearSolver{T}
    tril::SparseMatrixCSC{T,Int32}
    full::SparseMatrixCSC{T,Int32}
    tril_to_full_view::SubVector{T}

    control::Ma77Control{T}
    info::Ma77Info{T}

    mc68_control::Mc68Control
    mc68_info::Mc68Info

    order::Vector{Int32}
    keep::Vector{Ptr{Nothing}}

    opt::Ma77Options
    logger::MadNLPLogger
end

for (fdefault, fanalyse, ffactor, fsolve, ffinalise, fopen, finputv, finputr, typ) in [
    (:ma77_default_control_d, :ma77_analyse_d,
     :ma77_factor_d, :ma77_solve_d, :ma77_finalise_d,
     :ma77_open_d, :ma77_input_vars_d, :ma77_input_reals_d, Float64),
    (:ma77_default_control_s, :ma77_analyse_s,
     :ma77_factor_s, :ma77_solve_s, :ma77_finalise_s,
     :ma77_open_s, :ma77_input_vars_s, :ma77_input_reals_s, Float32),
    ]
    @eval begin
        ma77_default_control(control::Ma77Control{$typ}) = ccall(
            ($(string(fdefault)),libma77),
            Cvoid,
            (Ref{Ma77Control{$typ}},),
            control
        )
        ma77_open(
            n::Cint,fname1::String,fname2::String,fname3::String,fname4::String,
            keep::Vector{Ptr{Cvoid}},control::Ma77Control{$typ},info::Ma77Info{$typ}
        ) = ccall(
            ($(string(fopen)),libma77),
            Cvoid,
            (Cint,Ptr{Cchar},Ptr{Cchar},Ptr{Cchar},Ptr{Cchar},
             Ptr{Ptr{Cvoid}},Ref{Ma77Control{$typ}},Ref{Ma77Info{$typ}}),
            n,fname1,fname2,fname3,fname4,keep,control,info
        )
        ma77_input_vars(
            idx::Cint,nvar::Cint,list::Vector{Cint},
            keep::Vector{Ptr{Cvoid}},control::Ma77Control{$typ},info::Ma77Info{$typ}
        ) = ccall(
            ($(string(finputv)),libma77),
            Cvoid,
            (Cint,Cint,Ptr{Cint},
             Ptr{Ptr{Cvoid}},Ref{Ma77Control{$typ}},Ref{Ma77Info{$typ}}
             ),
                              idx,nvar,list,keep,control,info)
        ma77_input_reals(
            idx::Cint,length::Cint,reals::Vector{$typ},
            keep::Vector{Ptr{Cvoid}},control::Ma77Control{$typ},info::Ma77Info{$typ}
        ) = ccall(
            ($(string(finputr)),libma77),
            Cvoid,
            (Cint,Cint,Ptr{$typ},
             Ptr{Ptr{Cvoid}},Ref{Ma77Control{$typ}},Ref{Ma77Info{$typ}}),
            idx,length,reals,keep,control,info
        )
        ma77_analyse(
            order::Vector{Cint},
            keep::Vector{Ptr{Cvoid}},control::Ma77Control{$typ},info::Ma77Info{$typ}
        ) = ccall(
            ($(string(fanalyse)),libma77),
            Cvoid,
            (Ptr{Cint},Ptr{Ptr{Cvoid}},Ref{Ma77Control{$typ}},Ref{Ma77Info{$typ}}),
            order,keep,control,info
        )
        ma77_factor(
            posdef::Cint,keep::Vector{Ptr{Cvoid}},control::Ma77Control{$typ},info::Ma77Info{$typ},
            scale::Ptr{Nothing}
        ) = ccall(
            ($(string(ffactor)),libma77),
            Cvoid,
            (Cint,Ptr{Ptr{Cvoid}},Ref{Ma77Control{$typ}},Ref{Ma77Info{$typ}},Ptr{Nothing}),
            posdef,keep,control,info,scale
        )
        ma77_solve(
            job::Cint,nrhs::Cint,lx::Cint,x::Vector{$typ},
            keep::Vector{Ptr{Cvoid}},control::Ma77Control{$typ},info::Ma77Info{$typ},
            scale::Ptr{Nothing}
        ) = ccall(
            ($(string(fsolve)),libma77),
            Cvoid,
            (Cint,Cint,Cint,Ptr{$typ},
             Ptr{Ptr{Cvoid}},Ref{Ma77Control{$typ}},Ref{Ma77Info{$typ}},Ptr{Nothing}),
            job,nrhs,lx,x,keep,control,info,scale
        );
        ma77_finalize(
            keep::Vector{Ptr{Cvoid}},control::Ma77Control{$typ},info::Ma77Info{$typ}
        ) = ccall(
            ($(string(ffinalise)),libma77),
            Cvoid,
            (Ptr{Ptr{Cvoid}},Ref{Ma77Control{$typ}},Ref{Ma77Info{$typ}}),
            keep,control,info
        )
    end
end

function Ma77Solver(
    csc::SparseMatrixCSC{T,Int32};
    opt=Ma77Options(),logger=MadNLPLogger(),
) where T
    full,tril_to_full_view = get_tril_to_full(csc)
    order = Vector{Int32}(undef,csc.n)

    mc68_info = Mc68Info()
    mc68_control = get_mc68_default_control()

    keep = [C_NULL]

    mc68_control.f_array_in=1
    mc68_control.f_array_out=1
    mc68_order_i(Int32(opt.ma77_order),Int32(csc.n),csc.colptr,csc.rowval,order,mc68_control,mc68_info)

    info=Ma77Info{T}()
    control=Ma77Control{T}()
    ma77_default_control(control)
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

    ma77_open(Int32(full.n),".ma77_int", ".ma77_real", ".ma77_work", ".ma77_delay",
                keep,control,info)

    info.flag < 0 && throw(SymbolicException())

    for i=1:full.n
        ma77_input_vars(
            Int32(i),full.colptr[i+1]-full.colptr[i],
            _madnlp_unsafe_wrap(
                full.rowval,
                full.colptr[i+1]-full.colptr[i],
                full.colptr[i]
            ),
            keep,control,info);
        info.flag < 0 && throw(LinearSymbolicException())
    end

    ma77_analyse(order,keep,control,info)
    info.flag<0 && throw(SymbolicException())

    M = Ma77Solver{T}(csc,full,tril_to_full_view,
                   control,info,mc68_control,mc68_info,order,keep,opt,logger)
    finalizer(finalize,M)
    return M
end

function factorize!(M::Ma77Solver)
    M.full.nzval.=M.tril_to_full_view
    for i=1:M.full.n
        ma77_input_reals(
            Int32(i),M.full.colptr[i+1]-M.full.colptr[i],
            _madnlp_unsafe_wrap(
                M.full.nzval,
                M.full.colptr[i+1]-M.full.colptr[i],
                M.full.colptr[i]
            ),
            M.keep,M.control,M.info
        )
        M.info.flag < 0 && throw(FactorizationException())
    end
    ma77_factor(Int32(0),M.keep,M.control,M.info,C_NULL)
    M.info.flag < 0 && throw(FactorizationException())
    return M
end
function solve!(M::Ma77Solver{T},rhs::Vector{T}) where T
    ma77_solve(Int32(0),Int32(1),Int32(M.full.n),rhs,M.keep,M.control,M.info,C_NULL);
    M.info.flag < 0 && throw(SolveException())
    return rhs
end

is_inertia(::Ma77Solver) = true
function inertia(M::Ma77Solver)
    return (M.info.matrix_rank-M.info.num_neg,M.full.n-M.info.matrix_rank,M.info.num_neg)
end

finalize(M::Ma77Solver{T}) where T = ma77_finalize(M.keep,M.control,M.info)

function improve!(M::Ma77Solver)
    if M.control.u == M.opt.ma77_umax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.control.u = min(M.opt.ma77_umax,M.control.u^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.control.u)")
    return true
end

introduce(::Ma77Solver)="ma77"
input_type(::Type{Ma77Solver}) = :csc
default_options(::Type{Ma77Solver}) = Ma77Options()
is_supported(::Type{Ma77Solver},::Type{Float32}) = true
is_supported(::Type{Ma77Solver},::Type{Float64}) = true
