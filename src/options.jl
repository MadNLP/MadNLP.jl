# Options

parse_option(::Type{Module},str::String) = eval(Symbol(str))

function set_options!(opt::AbstractOptions, options)
    other_options = Dict{Symbol, Any}()
    for (key, val) in options
        if hasproperty(opt, key)
            T = fieldtype(typeof(opt), key)
            val isa T ? setproperty!(opt,key,val) :
                setproperty!(opt,key,parse_option(T,val))
        else
            other_options[key] = val
        end
    end
    return other_options
end

@kwdef mutable struct MadNLPOptions <: AbstractOptions
    # General options
    rethrow_error::Bool = true
    disable_garbage_collector::Bool = false
    blas_num_threads::Int = 1 
    linear_solver::Type = LapackCPUSolver
    iterator::Type = RichardsonIterator

    # Output options
    output_file::String = ""
    print_level::LogLevels = INFO
    file_print_level::LogLevels = INFO

    # Termination options
    tol::Float64 = 1e-8
    acceptable_tol::Float64 = 1e-6
    acceptable_iter::Int = 15
    diverging_iterates_tol::Float64 = 1e20
    max_iter::Int = 3000
    max_wall_time::Float64 = 1e6
    s_max::Float64 = 100.

    # NLP options
    kappa_d::Float64 = 1e-5
    fixed_variable_treatment::Type = MakeParameter
    equality_treatment::Type = EnforceEquality
    boudn_relax_factor::Float64 = 1e-8
    jacobian_constant::Bool = false
    hessian_constant::Bool = false
    kkt_system::Type = SparseKKTSystem
    hessian_approximation::Type = ExactHessian
    callback::Type = SparseCallback

    # initialization options
    dual_initialized::Bool = false
    inertia_correction_method::InertiaCorrectionMethod = INERTIA_AUTO
    constr_mult_init_max::Float64 = 1e3
    bound_push::Float64 = 1e-2
    bound_fac::Float64 = 1e-2
    nlp_scaling::Bool = true
    nlp_scaling_max_gradient::Float64 = 100.
    inertia_free_tol::Float64 = 0.

    # Hessian Perturbation
    min_hessian_perturbation::Float64 = 1e-20
    first_hessian_perturbation::Float64 = 1e-4
    max_hessian_perturbation::Float64 = 1e20
    perturb_inc_fact_first::Float64 = 1e2
    perturb_inc_fact::Float64 = 8.
    perturb_dec_fact::Float64 = 1/3
    jacobian_regularization_exponent::Float64 = 1/4
    jacobian_regularization_value::Float64 = 1e-8

    # restoration options
    soft_resto_pderror_reduction_factor::Float64 = 0.9999
    required_infeasibility_reduction::Float64 = 0.9

    # Line search
    obj_max_inc::Float64 = 5.
    kappha_soc::Float64 = 0.99
    max_soc::Int = 4
    alpha_min_frac::Float64 = 0.05
    s_theta::Float64 = 1.1
    s_phi::Float64 = 2.3
    eta_phi::Float64 = 1e-4
    kappa_soc::Float64 = 0.99
    gamma_theta::Float64 = 1e-5
    gamma_phi::Float64 = 1e-5
    delta::Float64 = 1
    kappa_sigma::Float64 = 1e10
    barrier_tol_factor::Float64 = 10.
    rho::Float64 = 1000.

    # Barrier
    mu_init::Float64 = 1e-1
    mu_min::Float64 = min(1e-4, tol ) / (barrier_tol_factor + 1) # by courtesy of Ipopt
    mu_superlinear_decrease_power::Float64 = 1.5
    tau_min::Float64 = 0.99
    mu_linear_decrease_factor::Float64 = .2
end

# smart option presets
function MadNLPOptions(nlp::AbstractNLPModel{T}) where T

    # if dense callback is defined, we use dense callback
    is_dense_callback =
        hasmethod(MadNLP.jac_dense!, Tuple{typeof(nlp), AbstractVector, AbstractMatrix}) &&
        hasmethod(MadNLP.hess_dense!, Tuple{typeof(nlp), AbstractVector, AbstractVector, AbstractMatrix})
    
    callback = is_dense_callback ? DenseCallback : SparseCallback

    # if dense callback is used, we use dense condensed kkt system
    kkt_system = is_dense_callback ? DenseCondensedKKTSystem : SparseKKTSystem

    # if dense kkt system, we use a dense linear solver
    linear_solver = is_dense_callback ? LapackCPUSolver : default_sparse_solver(nlp)

    tol = get_tolerance(T,kkt_system)
    
    return MadNLPOptions(
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        tol = tol,
    )
end

get_tolerance(::Type{T},::Type{KKT}) where {T, KKT} = 10^round(log10(eps(T))/2)
get_tolerance(::Type{T},::Type{SparseCondensedKKTSystem}) where T = 10^(round(log10(eps(T))/4))

function default_sparse_solver(nlp::AbstractNLPModel)
    if isdefined(Main, :MadNLPHSL)
        Main.MadNLPHSL.Ma27Solver
    elseif isdefined(Main, :MadNLPMumps)
        Main.MadNLPMumps.MumpsSolver
    else
        UmfpackSolver
    end
end

function check_option_sanity(options)
    is_kkt_dense = (options.kkt_system == DENSE_KKT_SYSTEM) || (options.kkt_system == DENSE_CONDENSED_KKT_SYSTEM)
    is_hess_approx_dense = (options.hessian_approximation == DENSE_BFGS) || (options.hessian_approximation == DENSE_DAMPED_BFGS)
    if input_type(options.linear_solver) == :csc && is_kkt_dense
        error("[options] Sparse Linear solver is not supported in dense mode.\n"*
              "Please use a dense linear solver or change `kkt_system` ")
    end
    if is_hess_approx_dense && !is_kkt_dense
        error("[options] DENSE_BFGS and DENSE_DAMPED_BFGS quasi-Newton approximations\n"*
              "require a dense KKT system (DENSE_KKT_SYSTEM or DENSE_CONDENSED_KKT_SYSTEM).")
    end
end

function print_ignored_options(logger,option_dict)
    @warn(logger,"The following options are ignored: ")
    for (key,val) in option_dict
        @warn(logger," - "*string(key))
    end
end

function load_options(nlp; options...)
    # Initiate interior-point options
    opt_ipm = MadNLPOptions(nlp)
    linear_solver_options = set_options!(opt_ipm, options)
    check_option_sanity(opt_ipm)
    # Initiate linear-solver options
    opt_linear_solver = default_options(opt_ipm.linear_solver)
    remaining_options = set_options!(opt_linear_solver, linear_solver_options)

    # Initiate logger
    logger = MadNLPLogger(
        print_level=opt_ipm.print_level,
        file_print_level=opt_ipm.file_print_level,
        file = opt_ipm.output_file == "" ? nothing : open(opt_ipm.output_file,"w+"),
    )
    @trace(logger,"Logger is initialized.")

    # Print remaning options (unsupported)
    if !isempty(remaining_options)
        print_ignored_options(logger, remaining_options)
    end
    return opt_ipm, opt_linear_solver, logger
end

