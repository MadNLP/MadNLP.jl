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

mutable struct Ma77Solver{T} <: AbstractLinearSolver{T}
    tril::SparseMatrixCSC{T,Int32}
    full::SparseMatrixCSC{T,Int32}
    tril_to_full_view::SubVector{T}

    control::ma77_control{T}
    info::ma77_info{T}

    mc68_control::mc68_control
    mc68_info::mc68_info

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
        ma77_default_control(control::ma77_control{$typ}
        ) = HSL.$fdefault(control)

        ma77_open(
            n::Cint,fname1::String,fname2::String,fname3::String,fname4::String,
            keep::Vector{Ptr{Cvoid}},control::ma77_control{$typ},info::ma77_info{$typ}
        ) = HSL.$fopen(n,fname1,fname2,fname3,fname4,keep,control,info)

        ma77_input_vars(
            idx::Cint,nvar::Cint,list::Vector{Cint},
            keep::Vector{Ptr{Cvoid}},control::ma77_control{$typ},info::ma77_info{$typ}
        ) = HSL.$finputv(idx,nvar,list,keep,control,info)

        ma77_input_reals(
            idx::Cint,length::Cint,reals::Vector{$typ},
            keep::Vector{Ptr{Cvoid}},control::ma77_control{$typ},info::ma77_info{$typ}
        ) = HSL.$finputr(idx,length,reals,keep,control,info)

        ma77_analyse(
            order::Vector{Cint},
            keep::Vector{Ptr{Cvoid}},control::ma77_control{$typ},info::ma77_info{$typ}
        ) = HSL.$fanalyse(order,keep,control,info)

        ma77_factor(
            posdef::Cint,keep::Vector{Ptr{Cvoid}},control::ma77_control{$typ},info::ma77_info{$typ},
            scale::Ptr{Nothing}
        ) = HSL.$ffactor(posdef,keep,control,info,scale)

        ma77_solve(
            job::Cint,nrhs::Cint,lx::Cint,x::Vector{$typ},
            keep::Vector{Ptr{Cvoid}},control::ma77_control{$typ},info::ma77_info{$typ},
            scale::Ptr{Nothing}
        ) = HSL.$fsolve(job,nrhs,lx,x,keep,control,info,scale)

        ma77_finalize(
            keep::Vector{Ptr{Cvoid}},control::ma77_control{$typ},info::ma77_info{$typ}
        ) = HSL.$ffinalise(keep,control,info)
    end
end

function Ma77Solver(
    csc::SparseMatrixCSC{T,Int32};
    opt=Ma77Options(),logger=MadNLPLogger(),
) where T
    full,tril_to_full_view = get_tril_to_full(csc)
    order = Vector{Int32}(undef,csc.n)

    mc68info = mc68_info()
    mc68control = mc68_control()
    HSL.mc68_default_control_i(mc68control)

    keep = [C_NULL]

    mc68control.f_array_in=1
    mc68control.f_array_out=1
    HSL.mc68_order_i(Int32(opt.ma77_order),Int32(csc.n),csc.colptr,csc.rowval,order,mc68control,mc68info)

    info=ma77_info{T}()
    control=ma77_control{T}()
    ma77_default_control(control)
    control.f_arrays = 1
    control.bits = 32
    control.file_size = opt.ma77_file_size
    control.maxstore = opt.ma77_maxstore
    control.print_level = -1

    control.buffer_lpage = (opt.ma77_buffer_lpage, opt.ma77_buffer_lpage)
    control.buffer_npage = (opt.ma77_buffer_npage, opt.ma77_buffer_npage)

    control.nemin = opt.ma77_nemin
    control.small = opt.ma77_small
    control.static_ = opt.ma77_static
    control.u = opt.ma77_u

    ma77_open(
        Int32(full.n),
        tempname(cleanup=false), tempname(cleanup=false), tempname(cleanup=false), tempname(cleanup=false),
        keep,control,info
    )

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
                   control,info,mc68control,mc68info,order,keep,opt,logger)
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

function finalize(M::Ma77Solver{T}) where T
    
    ma77_finalize(M.keep,M.control,M.info)
end

function improve!(M::Ma77Solver)
    if M.control.u == M.opt.ma77_umax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.control.u = min(M.opt.ma77_umax,M.control.u^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.control.u)")
    return true
end

introduce(::Ma77Solver)="ma77 v$(HSL.HSL_MA77_version())"
input_type(::Type{Ma77Solver}) = :csc
default_options(::Type{Ma77Solver}) = Ma77Options()
is_supported(::Type{Ma77Solver},::Type{Float32}) = true
is_supported(::Type{Ma77Solver},::Type{Float64}) = true
