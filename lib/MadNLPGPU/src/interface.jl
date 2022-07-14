
function CuInteriorPointSolver(nlp::AbstractNLPModel{T}; kwargs...) where T
    opt_ipm, opt_linear_solver, logger = MadNLP.load_options(; linear_solver=LapackGPUSolver, kwargs...)

    @assert is_supported(opt_ipm.linear_solver, T)
    KKTSystem = if (opt_ipm.kkt_system == MadNLP.SPARSE_KKT_SYSTEM) || (opt_ipm.kkt_system == MadNLP.SPARSE_UNREDUCED_KKT_SYSTEM)
        error("Sparse KKT system are currently not supported on CUDA GPU.\n" *
            "Please use `DENSE_KKT_SYSTEM` or `DENSE_CONDENSED_KKT_SYSTEM` instead.")
    elseif opt_ipm.kkt_system == MadNLP.DENSE_KKT_SYSTEM
        MT = CuMatrix{T}
        VT = CuVector{T}
        MadNLP.DenseKKTSystem{T, VT, MT}
    elseif opt_ipm.kkt_system == MadNLP.DENSE_CONDENSED_KKT_SYSTEM
        MT = CuMatrix{T}
        VT = CuVector{T}
        MadNLP.DenseCondensedKKTSystem{T, VT, MT}
    end
    return MadNLP.InteriorPointSolver{T,KKTSystem}(nlp, opt_ipm, opt_linear_solver; logger=logger)
end
