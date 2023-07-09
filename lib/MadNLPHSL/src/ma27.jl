ma27_default_icntl() = Int32[
    6,6,0,2139062143,1,32639,32639,32639,32639,14,9,8,8,9,10,32639,32639,32639,32689,24,11,9,8,9,10,0,0,0,0,0]
ma27_default_cntl(T)  = T[.1,1.0,0.,0.,0.]

@kwdef mutable struct Ma27Options <: AbstractOptions
    ma27_pivtol::Float64 = 1e-8
    ma27_pivtolmax::Float64 = 1e-4
    ma27_liw_init_factor::Float64 = 5.
    ma27_la_init_factor::Float64 =5.
    ma27_meminc_factor::Float64 =2.
end

struct Ma27Solver{T} <: AbstractLinearSolver{T}
    csc::SparseMatrixCSC{T,Int32}
    I::Vector{Int32}
    J::Vector{Int32}

    icntl::Vector{Int32}
    cntl::Vector{T}

    info::Vector{Int32}

    a::Vector{T}
    a_view::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}
    la::Int32
    ikeep::Vector{Int32}

    iw::Vector{Int32}
    liw::Int32
    iw1::Vector{Int32}
    nsteps::Vector{Int32}
    w::Vector{T}
    maxfrt::Vector{Int32}

    opt::Ma27Options
    logger::MadNLPLogger
end


for (fa, fb, fc, typ) in [
    (:ma27ad_,:ma27bd_,:ma27cd_,Float64),
    (:ma27a_,:ma27b_,:ma27c_,Float32)
    ]
    @eval begin
        ma27a!(
            n::Cint,nz::Cint,I::Vector{Cint},J::Vector{Cint},
            iw::Vector{Cint},liw::Cint,ikeep::Vector{Cint},iw1::Vector{Cint},
            nsteps::Vector{Cint},iflag::Cint,icntl::Vector{Cint},cntl::Vector{$typ},
            info::Vector{Cint},ops::$typ
        ) = ccall(
            ($(string(fa)),libhsl),
            Nothing,
            (Ref{Cint},Ref{Cint},Ptr{Cint},Ptr{Cint},
             Ptr{Cint},Ref{Cint},Ptr{Cint},Ptr{Cint},
             Ptr{Cint},Ref{Cint},Ptr{Cint},Ptr{$typ},
             Ptr{Cint},Ref{$typ}),
            n,nz,I,J,iw,liw,ikeep,iw1,nsteps,iflag,icntl,cntl,info,ops
        )

        ma27b!(
            n::Cint,nz::Cint,I::Vector{Cint},J::Vector{Cint},
            a::Vector{$typ},la::Cint,iw::Vector{Cint},liw::Cint,
            ikeep::Vector{Cint},nsteps::Vector{Cint},maxfrt::Vector{Cint},iw1::Vector{Cint},
            icntl::Vector{Cint},cntl::Vector{$typ},info::Vector{Cint}
        ) = ccall(
            ($(string(fb)),libhsl),
            Nothing,
            (Ref{Cint},Ref{Cint},Ptr{Cint},Ptr{Cint},
             Ptr{$typ},Ref{Cint},Ptr{Cint},Ref{Cint},
             Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},
             Ptr{Cint},Ptr{$typ},Ptr{Cint}),
            n,nz,I,J,a,la,iw,liw,ikeep,nsteps,maxfrt,iw1,icntl,cntl,info
        )

        ma27c!(
            n::Cint,a::Vector{$typ},la::Cint,iw::Vector{Cint},
            liw::Cint,w::Vector{$typ},maxfrt::Vector{Cint},rhs::Vector{$typ},
            iw1::Vector{Cint},nsteps::Vector{Cint},icntl::Vector{Cint},
            info::Vector{Cint}
        ) = ccall(
            ($(string(fc)),libhsl),
            Nothing,
            (Ref{Cint},Ptr{$typ},Ref{Cint},Ptr{Cint},
             Ref{Cint},Ptr{$typ},Ptr{Cint},Ptr{$typ},
             Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
            n,a,la,iw,liw,w,maxfrt,rhs,iw1,nsteps,icntl,info
        )
    end
end

function Ma27Solver(csc::SparseMatrixCSC{T};
    opt=Ma27Options(),logger=MadNLPLogger(),
) where T
    I,J = findIJ(csc)
    nz=Int32(nnz(csc))

    liw= Int32(2*(2*nz+3*csc.n+1))
    iw = Vector{Int32}(undef,liw)
    ikeep= Vector{Int32}(undef,3*csc.n)
    iw1  = Vector{Int32}(undef,2*csc.n)
    nsteps=Int32[1]
    iflag =Int32(0)

    icntl= ma27_default_icntl()
    cntl = ma27_default_cntl(T)
    icntl[1:2] .= 0
    cntl[1] = opt.ma27_pivtol

    info = Vector{Int32}(undef,20)
    ma27a!(Int32(csc.n),nz,I,J,iw,liw,ikeep,iw1,nsteps,Int32(0),icntl,cntl,info,zero(T))
    info[1]<0 && throw(SymbolicException())

    la = ceil(Int32,max(nz,opt.ma27_la_init_factor*info[5]))
    a = Vector{T}(undef,la)
    a_view = view(a,1:nnz(csc)) # _madnlp_unsafe_wrap is not used because we may resize a
    liw= ceil(Int32,opt.ma27_liw_init_factor*info[6])
    resize!(iw,liw)
    maxfrt=Int32[1]

    return Ma27Solver{T}(csc,I,J,icntl,cntl,info,a,a_view,la,ikeep,iw,liw,
                     iw1,nsteps,Vector{T}(),maxfrt,opt,logger)
end


function factorize!(M::Ma27Solver)
    M.a_view.=M.csc.nzval
    while true
        ma27b!(Int32(M.csc.n),Int32(nnz(M.csc)),M.I,M.J,M.a,M.la,
                M.iw,M.liw,M.ikeep,M.nsteps,M.maxfrt,
                M.iw1,M.icntl,M.cntl,M.info)
        if M.info[1] == -3
            M.liw = ceil(Int32,M.opt.ma27_meminc_factor*M.liw)
            resize!(M.iw, M.liw)
            @debug(M.logger,"Reallocating memory: liw ($(M.liw))")
        elseif M.info[1] == -4
            M.la = ceil(Int32,M.opt.ma27_meminc_factor*M.la)
            resize!(M.a,M.la)
            @debug(M.logger,"Reallocating memory: la ($(M.la))")
        elseif M.info[1] < 0
            throw(FactorizationException())
        else
            break
        end
    end
    return M
end

function solve!(M::Ma27Solver{T},rhs::Vector{T}) where T
    length(M.w)<M.maxfrt[1] && resize!(M.w,M.maxfrt[1])
    length(M.iw1)<M.nsteps[1] && resize!(M.iw1,M.nsteps[1])
    ma27c!(Int32(M.csc.n),M.a,M.la,M.iw,M.liw,M.w,M.maxfrt,rhs,
            M.iw1,M.nsteps,M.icntl,M.info)
    M.info[1]<0 && throw(SolveException())
    return rhs
end

is_inertia(::Ma27Solver) = true
function inertia(M::Ma27Solver)
    dim = M.csc.n
    rank = (Int(M.info[1])==3) ? Int(M.info[2]) : dim
    neg = Int(M.info[15])

    return (rank-neg,dim-rank,neg) 
end

function improve!(M::Ma27Solver)
    if M.cntl[1] == M.opt.ma27_pivtolmax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.cntl[1] = min(M.opt.ma27_pivtolmax,M.cntl[1]^.75)
    @debug(M.logger,"improved quality: pivtol = $(M.cntl[1])")
    return true
end

introduce(::Ma27Solver)="ma27"
input_type(::Type{Ma27Solver}) = :csc
default_options(::Type{Ma27Solver}) = Ma27Options()
is_supported(::Type{Ma27Solver},::Type{Float32}) = true
is_supported(::Type{Ma27Solver},::Type{Float64}) = true
