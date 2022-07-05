const umfpack_default_ctrl = copy(UMFPACK.umf_ctrl)
const umfpack_default_info = copy(UMFPACK.umf_info)

@kwdef mutable struct UmfpackOptions <: AbstractOptions
    umfpack_pivtol::Float64 = 1e-4
    umfpack_pivtolmax::Float64 = 1e-1
    umfpack_sym_pivtol::Float64 = 1e-3
    umfpack_block_size::Float64 = 16
    umfpack_strategy::Float64 = 2.
end

mutable struct UmfpackSolver{T} <: AbstractLinearSolver{T}
    inner::UMFPACK.UmfpackLU
    tril::SparseMatrixCSC{T}
    full::SparseMatrixCSC{T}
    tril_to_full_view::SubVector{T}

    p::Vector{T}

    tmp::Vector{Ptr{Cvoid}}
    ctrl::Vector{T}
    info::Vector{T}

    opt::UmfpackOptions
    logger::Logger
end


for (numeric,solve,T) in (
    (:umfpack_di_numeric, :umfpack_di_solve, Float64),
    (:umfpack_si_numeric, :umfpack_si_solve, Float32),
    )
    @eval begin 
        umfpack_numeric(
            colptr::Vector{Int32},rowval::Vector{Int32},
            nzval::Vector{$T},symbolic::Ptr{Nothing},
            tmp::Vector{Ptr{Nothing}},ctrl::Vector{$T},
            info::Vector{$T}) = ccall(
                ($(string(numeric)),:libumfpack),
                Int32,
                (Ptr{Int32},Ptr{Int32},Ptr{$T},Ptr{Cvoid},Ptr{Cvoid},
                 Ptr{$T},Ptr{$T}),
                colptr,rowval,nzval,symbolic,tmp,ctrl,info)
        umfpack_solve(
            typ,colptr::Vector{Int32},rowval::Vector{Int32},
            nzval::Vector{$T},x::Vector{$T},b::Vector{$T},
            numeric,ctrl::Vector{$T},info::Vector{$T}) = ccall(
                ($(string(solve)),:libumfpack),
                Int32,
                (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{$T},Ptr{$T},
                 Ptr{$T}, Ptr{Cvoid}, Ptr{$T},Ptr{$T}),
                typ,colptr,rowval,nzval,x,b,numeric,ctrl,info)
    end
end



function UmfpackSolver(
    csc::SparseMatrixCSC{T};
    option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    opt=UmfpackOptions(),logger=Logger(),
    kwargs...) where T

    set_options!(opt,option_dict,kwargs)

    p = Vector{T}(undef,csc.n)
    full,tril_to_full_view = get_tril_to_full(csc)

    full.colptr.-=1; full.rowval.-=1

    inner = UMFPACK.UmfpackLU(C_NULL,C_NULL,full.n,full.n,full.colptr,full.rowval,full.nzval,0)
    UMFPACK.finalizer(UMFPACK.umfpack_free_symbolic,inner)
    UMFPACK.umfpack_symbolic!(inner)
    ctrl = copy(umfpack_default_ctrl)
    info = copy(umfpack_default_info)
    ctrl[4]=opt.umfpack_pivtol
    ctrl[12]=opt.umfpack_sym_pivtol
    ctrl[5]=opt.umfpack_block_size
    ctrl[6]=opt.umfpack_strategy

    tmp = Vector{Ptr{Cvoid}}(undef, 1)

    return UmfpackSolver(inner,csc,full,tril_to_full_view,p,tmp,ctrl,info,opt,logger)
end

function factorize!(M::UmfpackSolver)
    UMFPACK.umfpack_free_numeric(M.inner)
    M.full.nzval.=M.tril_to_full_view
    status = umfpack_numeric(M.inner.colptr,M.inner.rowval,M.inner.nzval,M.inner.symbolic,M.tmp,M.ctrl,M.info)
    M.inner.numeric = M.tmp[]

    M.inner.status = status
    return M
end
function solve!(M::UmfpackSolver{T},rhs::Vector{T}) where T
    status = umfpack_solve(1,M.inner.colptr,M.inner.rowval,M.inner.nzval,M.p,rhs,M.inner.numeric,M.ctrl,M.info)
    rhs .= M.p
    return rhs
end
is_inertia(::UmfpackSolver) = false
inertia(M::UmfpackSolver) = throw(InertiaException())
input_type(::Type{UmfpackSolver}) = :csc

function improve!(M::UmfpackSolver)
    if M.ctrl[4] == M.opt.umfpack_pivtolmax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.ctrl[4] = min(M.opt.umfpack_pivtolmax,M.ctrl[4]^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.ctrl[4])")
    return true

    return false
end
introduce(::UmfpackSolver)="umfpack"
is_supported(::Type{UmfpackSolver},::Type{Float64}) = true
