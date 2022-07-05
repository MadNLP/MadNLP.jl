
function CuInteriorPointSolver(nlp::AbstractNLPModel{T};
    option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(), kwargs...
) where T
    opt = MadNLP.Options(linear_solver=LapackGPUSolver)
    MadNLP.set_options!(opt,option_dict,kwargs)
    MadNLP.check_option_sanity(opt)

    KKTSystem = if (opt.kkt_system == MadNLP.SPARSE_KKT_SYSTEM) || (opt.kkt_system == MadNLP.SPARSE_UNREDUCED_KKT_SYSTEM)
        error("Sparse KKT system are currently not supported on CUDA GPU.\n" *
            "Please use `DENSE_KKT_SYSTEM` or `DENSE_CONDENSED_KKT_SYSTEM` instead.")
    elseif opt.kkt_system == MadNLP.DENSE_KKT_SYSTEM
        MT = CuMatrix{T}
        VT = CuVector{T}
        MadNLP.DenseKKTSystem{T, VT, MT}
    elseif opt.kkt_system == MadNLP.DENSE_CONDENSED_KKT_SYSTEM
        MT = CuMatrix{T}
        VT = CuVector{T}
        MadNLP.DenseCondensedKKTSystem{T, VT, MT}
    end
    return MadNLP.InteriorPointSolver{T,KKTSystem}(nlp, opt; option_linear_solver=option_dict)
end
