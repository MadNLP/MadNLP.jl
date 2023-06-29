@kwdef mutable struct RFSolverOptions <: MadNLP.AbstractOptions
    symbolic_analysis::Symbol = :klu
    fast_mode::Bool = true
    factorization_algo::CUSOLVER.cusolverRfFactorization_t = CUSOLVER.CUSOLVERRF_FACTORIZATION_ALG0
    triangular_solve_algo::CUSOLVER.cusolverRfTriangularSolve_t = CUSOLVER.CUSOLVERRF_TRIANGULAR_SOLVE_ALG1
end


const CuSubVector{T} = SubArray{T, 1, CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer}, Tuple{CUDA.CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}}, false}

mutable struct RFSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing,CUSOLVERRF.RFLowLevel}

    tril::CUSPARSE.CuSparseMatrixCSC{T}
    full::CUSPARSE.CuSparseMatrixCSR{T}
    tril_to_full_view::CuSubVector{T}

    opt::RFSolverOptions
    logger::MadNLP.MadNLPLogger
end

function RFSolver(
    csc::CUSPARSE.CuSparseMatrixCSC{Float64};
    opt=RFSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
)
    n, m = size(csc)
    @assert n == m

    full,tril_to_full_view = MadNLP.get_tril_to_full(csc)
    
    return RFSolver{Float64}(
        nothing, csc, full, tril_to_full_view,
        opt, logger
    )
end

function MadNLP.factorize!(M::RFSolver)
    copyto!(M.full.nzVal, M.tril_to_full_view)
    if M.inner == nothing
        sym_lu = CUSOLVERRF.klu_symbolic_analysis(M.full)
        M.inner = CUSOLVERRF.RFLowLevel(
            sym_lu;
            fast_mode=M.opt.fast_mode,
            factorization_algo=M.opt.factorization_algo,
            triangular_algo=M.opt.triangular_solve_algo,
        )
    end
    CUSOLVERRF.rf_refactor!(M.inner, M.full)
    return M
end

function MadNLP.solve!(M::RFSolver{Float64}, x)
    CUSOLVERRF.rf_solve!(M.inner, x)
    return x
end

MadNLP.input_type(::Type{RFSolver}) = :csc
MadNLP.default_options(::Type{RFSolver}) = RFSolverOptions()
MadNLP.is_inertia(M::RFSolver) = true
MadNLP.inertia(M::RFSolver) = (size(M.full,1),0,0)
MadNLP.improve!(M::RFSolver) = false
MadNLP.is_supported(::Type{RFSolver},::Type{Float32}) = false
MadNLP.is_supported(::Type{RFSolver},::Type{Float64}) = true
MadNLP.introduce(M::RFSolver) = "cuSolverRF"
