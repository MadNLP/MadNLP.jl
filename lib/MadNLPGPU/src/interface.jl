
function CuInteriorPointSolver(nlp::AbstractNLPModel{T};
    option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(), kwargs...
) where T
    # Initiate interior-point options
    opt = MadNLP.IPMOptions(linear_solver=LapackGPUSolver)
    linear_solver_options = set_options!(opt, kwargs)
    MadNLP.check_option_sanity(opt)

    # Initiate linear-solver options
    @assert is_supported(opt.linear_solver, T)
    opt_linear_solver = default_options(opt.linear_solver)
    remaining_options = set_options!(opt_linear_solver, linear_solver_options)

    # Initiate logger
    logger = MadNLP.Logger(
        print_level=opt.print_level,
        file_print_level=opt.file_print_level,
        file = opt.output_file == "" ? nothing : open(opt.output_file,"w+"),
    )
    MadNLP.@trace(logger,"Logger is initialized.")

    # Print remaning options (unsupported)
    if !isempty(remaining_options)
        MadNLP.print_ignored_options(logger, remaining_options)
    end

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
    return MadNLP.InteriorPointSolver{T,KKTSystem}(nlp, opt, opt_linear_solver; logger=logger)
end
