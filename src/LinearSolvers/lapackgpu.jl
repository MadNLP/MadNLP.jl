module LapackGPU

import ..MadNLP:
    @kwdef, Logger, @debug, @warn, @error,
    CUBLAS, CUSOLVER, CuVector, CuMatrix,
    AbstractOptions, AbstractLinearSolver, set_options!,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, LapackCPU, tril_to_full!

const INPUT_MATRIX_TYPE = :dense

@enum(Algorithms::Int, BUNCHKAUFMAN = 1, LU = 2, QR = 3)
@kwdef mutable struct Options <: AbstractOptions
    lapackgpu_algorithm::Algorithms = BUNCHKAUFMAN
end

mutable struct Solver <: AbstractLinearSolver
    dense::Matrix{Float64}
    fact::CuMatrix{Float64}
    rhs::CuVector{Float64}
    work::CuVector{Float64}
    lwork
    info::CuVector{Int32}
    etc::Dict{Symbol,Any} # throw some algorithm-specific things here
    opt::Options
    logger::Logger
end

function Solver(dense::Matrix{Float64};
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Options(),logger=Logger(),
                kwargs...)

    set_options!(opt,option_dict,kwargs...)
    fact = CuMatrix{Float64}(undef,size(dense))
    rhs = CuVector{Float64}(undef,size(dense,1))
    work  = CuVector{Float64}(undef, 1)
    lwork = Int32[1]
    info = CuVector{Int32}(undef,1)
    etc = Dict{Symbol,Any}()

    return Solver(dense,fact,rhs,work,lwork,info,etc,opt,logger)
end

function factorize!(M::Solver)
    if M.opt.lapackgpu_algorithm == BUNCHKAUFMAN
        factorize_bunchkaufman!(M)
    elseif M.opt.lapackgpu_algorithm == LU
        factorize_lu!(M)
    elseif M.opt.lapackgpu_algorithm == QR
        factorize_qr!(M)
    else
        error(LOGGER,"Invalid lapackgpu_algorithm")
    end
end
function solve!(M::Solver,x)
    if M.opt.lapackgpu_algorithm == BUNCHKAUFMAN
        solve_bunchkaufman!(M,x)
    elseif M.opt.lapackgpu_algorithm == LU
        solve_lu!(M,x)
    elseif M.opt.lapackgpu_algorithm == QR
        solve_qr!(M,x)
    else
        error(LOGGER,"Invalid lapackgpu_algorithm")
    end
end

is_inertia(M::Solver) = M.opt.lapackgpu_algorithm == BUNCHKAUFMAN
inertia(M::Solver) = LapackCPU.inertia(M.etc[:fact_cpu],M.etc[:ipiv_cpu],M.info[])
improve!(M::Solver) = false
introduce(M::Solver) = "Lapack-GPU ($(M.opt.lapackgpu_algorithm))"

function factorize_bunchkaufman!(M::Solver)
    haskey(M.etc,:ipiv) || (M.etc[:ipiv] = CuVector{Int32}(undef,size(M.dense,1)))

    copyto!(M.fact,M.dense)
    CUSOLVER.cusolverDnDsytrf_bufferSize(
        CUSOLVER.dense_handle(),Int32(size(M.fact,1)),M.fact,Int32(size(M.fact,2)),M.lwork)
    length(M.work) < M.lwork[] && resize!(M.work,Int(M.lwork[]))
    CUSOLVER.cusolverDnDsytrf(
        CUSOLVER.dense_handle(),CUBLAS.CUBLAS_FILL_MODE_LOWER,
        Int32(size(M.fact,1)),M.fact,Int32(size(M.fact,2)),
        M.etc[:ipiv],M.work,M.lwork[],M.info)

    # need to send the factorization back to cpu to call mkl sytrs --------------
    haskey(M.etc,:fact_cpu) || (M.etc[:fact_cpu] = Matrix{Float64}(undef,size(M.dense)))
    haskey(M.etc,:ipiv_cpu) || (M.etc[:ipiv_cpu] = Vector{Int}(undef,size(M.dense,1)))
    copyto!(M.etc[:fact_cpu],M.fact)
    copyto!(M.etc[:ipiv_cpu],M.etc[:ipiv])
    # ---------------------------------------------------------------------------
    return M
end

function solve_bunchkaufman!(M::Solver,x)
    # It seems that Nvidia haven't implement sytrs yet -------------------
    # copyto!(M.rhs,x)
    # CUSOLVER.cusolverDnDsytrs_bufferSize(
    #     CUSOLVER.dense_handle(),CUBLAS.CUBLAS_FILL_MODE_LOWER,
    #     Int32(size(M.fact,1)),Int32(1),M.fact,Int32(size(M.fact,2)),
    #     M.etc[:ipiv],M.rhs,Int32(length(M.rhs)),M.lwork)
    # length(M.work) < M.lwork[] && resize!(M.work,Int(M.lwork[]))
    # CUSOLVER.cusolverDnDsytrs(
    #     CUSOLVER.dense_handle(),CUBLAS.CUBLAS_FILL_MODE_LOWER,
    #     Int32(size(M.fact,1)),Int32(1),M.fact,Int32(size(M.fact,2)),
    #     M.etc[:ipiv],M.rhs,Int32(length(M.rhs)),M.work,M.lwork[],M.info)
    # copyto!(x,M.rhs)
    # --------------------------------------------------------------------

    ccall(
        (:dsytrs_64_,"libopenblas64_"), # MKL doesn't work for some reason...
        Cvoid,
        (Ref{Cchar},Ref{Int},Ref{Int},Ptr{Cdouble},Ref{Int},Ptr{Int},Ptr{Cdouble},Ref{Int},Ptr{Int}),
        'L',size(M.fact,1),1,M.etc[:fact_cpu],size(M.fact,2),M.etc[:ipiv_cpu],x,length(x),[1])
    
    return x
end

function factorize_lu!(M::Solver)
    haskey(M.etc,:ipiv) || (M.etc[:ipiv] = CuVector{Int32}(undef,size(M.dense,1)))
    tril_to_full!(M.dense)
    copyto!(M.fact,M.dense)
    CUSOLVER.cusolverDnDgetrf_bufferSize(
        CUSOLVER.dense_handle(),Int32(size(M.fact,1)),Int32(size(M.fact,2)),
        M.fact,Int32(size(M.fact,2)),M.lwork)
    length(M.work) < M.lwork[] && resize!(M.work,Int(M.lwork[]))
    CUSOLVER.cusolverDnDgetrf(
        CUSOLVER.dense_handle(),Int32(size(M.fact,1)),Int32(size(M.fact,2)),
        M.fact,Int32(size(M.fact,2)),M.work,M.etc[:ipiv],M.info)
    return M
end

function solve_lu!(M::Solver,x)
    copyto!(M.rhs,x)
    CUSOLVER.cusolverDnDgetrs(
        CUSOLVER.dense_handle(),CUBLAS.CUBLAS_OP_N,
        Int32(size(M.fact,1)),Int32(1),M.fact,Int32(size(M.fact,2)),
        M.etc[:ipiv],M.rhs,Int32(length(M.rhs)),M.info)
    copyto!(x,M.rhs)
    return x
end

function factorize_qr!(M::Solver)
    haskey(M.etc,:tau) || (M.etc[:tau] = CuVector{Float64}(undef,size(M.dense,1)))
    haskey(M.etc,:one) || (M.etc[:one] = ones(1))
    tril_to_full!(M.dense)
    copyto!(M.fact,M.dense)
    CUSOLVER.cusolverDnDgeqrf_bufferSize(CUSOLVER.dense_handle(),Int32(size(M.fact,1)),Int32(size(M.fact,2)),M.fact,Int32(size(M.fact,2)),M.lwork)
    length(M.work) < M.lwork[] && resize!(M.work,Int(M.lwork[]))
    CUSOLVER.cusolverDnDgeqrf(CUSOLVER.dense_handle(),Int32(size(M.fact,1)),Int32(size(M.fact,2)),M.fact,Int32(size(M.fact,2)),M.etc[:tau],M.work,M.lwork[],M.info)
    return M
end

function solve_qr!(M::Solver,x)
    copyto!(M.rhs,x)
    CUSOLVER.cusolverDnDormqr_bufferSize(CUSOLVER.dense_handle(),CUBLAS.CUBLAS_SIDE_LEFT,CUBLAS.CUBLAS_OP_T,
                                         Int32(size(M.fact,1)),Int32(1),Int32(length(M.etc[:tau])),M.fact,Int32(size(M.fact,2)),M.etc[:tau],M.rhs,Int32(length(M.rhs)),M.lwork)
    length(M.work) < M.lwork[] && resize!(M.work,Int(M.lwork[]))
    CUSOLVER.cusolverDnDormqr(CUSOLVER.dense_handle(),CUBLAS.CUBLAS_SIDE_LEFT,CUBLAS.CUBLAS_OP_T,
                              Int32(size(M.fact,1)),Int32(1),Int32(length(M.etc[:tau])),M.fact,Int32(size(M.fact,2)),M.etc[:tau],M.rhs,Int32(length(M.rhs)),M.work,M.lwork[],M.info)
    CUBLAS.cublasDtrsm_v2(CUBLAS.handle(),CUBLAS.CUBLAS_SIDE_LEFT,CUBLAS.CUBLAS_FILL_MODE_UPPER,CUBLAS.CUBLAS_OP_N,CUBLAS.CUBLAS_DIAG_NON_UNIT,
                          Int32(size(M.fact,1)),Int32(1),M.etc[:one],M.fact,Int32(size(M.fact,2)),M.rhs,Int32(length(M.rhs)))
    copyto!(x,M.rhs)
    return x
end

end # module



# forgiving names
lapackgpu = LapackGPU
LAPACKGPU = LapackGPU
