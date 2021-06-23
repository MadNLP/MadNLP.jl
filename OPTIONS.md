## MadNLP Options
### Interior Point Solver Options
- `linear_solver::Module = DefaultLinearSolver`:\
    Linear solver used for solving primal-dual system. Valid values are: {`MadNLPUmfpack`, `MadNLPMumps`, `MadNLPPardisoMKL`, `MadNLPMa27`, `MadNLPMa57`, `MadNLPMa77`, `MadNLPMa86`, `MadNLPMa97`, `MadNLPPardiso`, `MadNLPSchur`, `MadNLPSchwarz`, `MadNLPLapackCPU`, `MadNLPLapackGPU`} (some may require using extension packages). The selected solver should be properly built in the build procedure. See [README.md](https://github.com/sshin23/MadNLP.jl) file.
- `iterator::Module = Richardson `\
    Iterator used for iterative refinement. Valid values are: {`MadNLPRichardson`,`MadNLPKrylov`}.
    - `Richardson` uses [Richardson iteration](https://en.wikipedia.org/wiki/Modified_Richardson_iteration)
    - `Krylov` uses [restarted Generalized Minimal Residual](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method) method implemented in [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl).
- `blas_num_threads::Int = 1`\
    Number of threads used for BLAS routines. Valid range is ``[1,\infty)``.
- `disable_garbage_collector::Bool = false `\
    If `true`, Julia garbage collector is temporarily disabled while solving the problem, and then enabled back once the solution is complete.
- `rethrow_error::Bool = true `\
    If `false`, any internal error thrown by `MadNLP` and interruption exception (triggered by the user via `^C`) is caught, and not rethrown. If an error is caught, the solver terminates with an error message.
- `print_level::LogLevels = INFO`\
    `stdout` print level. Any message with level less than `print_level` is not printed on `stdout`. Valid values are: `MadNLP`.{`TRACE`, `DEBUG`, `INFO`, `NOTICE`, `WARN`, `ERROR`}.
- `output_file::String = INFO`\
    If not `""`, the output log is teed to the file at the path specified in `output_file`.
- `file_print_level::LogLevels = TRACE`\
    File print level; any message with level less than `file_print_level` is not printed on the file specified in `output_file`. Valid values are: `MadNLP`.{`TRACE`, `DEBUG`, `INFO`, `NOTICE`, `WARN`, `ERROR`}.
- `tol::Float64 = 1e-8`\
    Termination tolerance. The solver terminates if the scaled primal, dual, complementary infeasibility is less than `tol`. Valid range is ``(0,\infty)``.
- `acceptable_tol::Float64 = 1e-6`\
    Acceptable tolerance. The solver terminates if the scaled primal, dual, complementary infeasibility is less than `acceptable_tol`, for `acceptable_iter` consecutive interior point iteration steps.
- `acceptable_iter::Int = 15`\
    Acceptable iteration tolerance. Valid rage is ``[1,\infty)``.
- `diverging_iterates_tol::Float64 = 1e20`\
    Diverging iteration tolerance. The solver terminates with exit symbol `:Diverging_Iterates` if the NLP error is greater than `diverging_iterates_tol`.
- `max_iter::Int = 3000`\
    Maximum number of interior point iterations. The solver terminates with exit symbol `:Maximum_Iterations_Exceeded` if the interior point iteration count exceeds `max_iter`.
- `max_wall_time::Float64 = 1e6`\
    Maximum wall time for interior point solver. The solver terminates with exit symbol `:Maximum_WallTime_Exceeded` if the total solver wall time exceeds `max_wall_time`.
- `fixed_variable_treatment::FixedVariableTreatments = MAKE_PARAMETER`\
    Valid values are: `MadNLP`.{`RELAX_BOUND`,`MAKE_PARAMETER`}.
- `jacobian_constant::Bool = false`\
    If `true`, constraint Jacobian is only evaluated once and reused.
- `hessian_constant::Bool = false`\
    If `true`, Lagrangian Hessian is only evaluated once and reused.
- `reduced_system::Bool = true`\
    If `true`, the primal-dual system is formulated as in Greif (2014).
- `inertia_correction_method::InertiaCorrectionMethods = INERTIA_AUTO`\
    Valid values are: `MadNLP`.{`INERTIA_AUTO`,`INERTIA_BASED`, `INERTIA_FREE`}.
    - `INERTIA_BASED` uses the strategy in Ipopt.
    - `INERTIA_FREE` uses the strategy in Chiang (2016).
    - `INERTIA_AUTO` uses `INERTIA_BASED` if inertia information is available and uses `INERTIA_FREE` otherwise.
- `s_max::Float64 = 100.`
- `kappa_d::Float64 = 1e-5`
- `constr_mult_init_max::Float64 = 1e3`
- `bound_push::Float64 = 1e-2`
- `bound_fac::Float64 = 1e-2`
- `nlp_scaling_max_gradient::Float64 = 100.`
- `inertia_free_tol::Float64 = 0.`
- `min_hessian_perturbation::Float64 = 1e-20`
- `first_hessian_perturbation::Float64 = 1e-4`
- `max_hessian_perturbation::Float64 = 1e20`
- `perturb_inc_fact_first::Float64 = 1e2`
- `perturb_inc_fact::Float64 = 8.`
- `perturb_dec_fact::Float64 = 1/3`
- `jacobian_regularization_exponent::Float64 = 1/4`
- `jacobian_regularization_value::Float64 = 1e-8`
- `soft_resto_pderror_reduction_factor::Float64 = 0.9999`
- `required_infeasibility_reduction::Float64 = 0.9`
- `obj_max_inc::Float64 = 5.`
- `kappha_soc::Float64 = 0.99`
- `max_soc::Int = 4`
- `alpha_min_frac::Float64 = 0.05`
- `s_theta::Float64 = 1.1`
- `s_phi::Float64 = 2.3`
- `eta_phi::Float64 = 1e-4`
- `kappa_soc::Float64 = 0.99`
- `gamma_theta::Float64 = 1e-5`
- `gamma_phi::Float64 = 1e-5`
- `delta::Float64 = 1`
- `kappa_sigma::Float64 = 1e10`
- `barrier_tol_factor::Float64 = 10.`
- `rho::Float64 = 1000.`
- `mu_init::Float64 = 1e-1`
- `mu_min::Float64 = 1e-9`
- `mu_superlinear_decrease_power::Float64 = 1.5`
- `tau_min::Float64 = 0.99`
- `mu_linear_decrease_factor::Float64 = .2`

### Linear Solver Options
Linear solver options are specific to the linear solver chosen at `linear_solver` option. Irrelevant options are ignored and a warning message is printed.
#### Ma27 (requires `MadNLPHSL`)
- `ma27_pivtol::Float64 = 1e-8`
- `ma27_pivtolmax::Float64 = 1e-4`
- `ma27_liw_init_factor::Float64 = 5.`
- `ma27_la_init_factor::Float64 = 5.`
- `ma27_meminc_factor::Float64 = 2.`

#### Ma57 (requires `MadNLPHSL`)
- `ma57_pivtol::Float64 = 1e-8`
- `ma57_pivtolmax::Float64 = 1e-4`
- `ma57_pre_alloc::Float64 = 1.05`
- `ma57_pivot_order::Int = 5`
- `ma57_automatic_scaling::Bool = false`
- `ma57_block_size::Int = 16`
- `ma57_node_amalgamation::Int = 16`
- `ma57_small_pivot_flag::Int = 0`

#### Ma77 (requires `MadNLPHSL`)
- `ma77_buffer_lpage::Int = 4096`
- `ma77_buffer_npage::Int = 1600`
- `ma77_file_size::Int = 2097152`
- `ma77_maxstore::Int = 0`
- `ma77_nemin::Int = 8`
- `ma77_order::Ma77.Ordering = Ma77.METIS`
- `ma77_print_level::Int = -1`
- `ma77_small::Float64 = 1e-20`
- `ma77_static::Float64 = 0.`
- `ma77_u::Float64 = 1e-8`
- `ma77_umax::Float64 = 1e-4`

#### Ma86 (requires `MadNLPHSL`)
- `ma86_num_threads::Int = 1`
- `ma86_print_level::Float64 = -1`
- `ma86_nemin::Int = 32`
- `ma86_order::Ma86.Ordering = Ma86.METIS`
- `ma86_scaling::Ma86.Scaling = Ma86.SCALING_NONE`
- `ma86_small::Float64 = 1e-20`
- `ma86_static::Float64 = 0.`
- `ma86_u::Float64 = 1e-8`
- `ma86_umax::Float64 = 1e-4`

#### Ma97 (requires `MadNLPHSL`)
- `ma97_num_threads::Int = 1`
- `ma97_print_level::Int = -1`
- `ma97_nemin::Int = 8`
- `ma97_order::Ma97.Ordering = Ma97.METIS`
- `ma97_scaling::Ma97.Scaling = Ma97.SCALING_NONE`
- `ma97_small::Float64 = 1e-20`
- `ma97_u::Float64 = 1e-8`
- `ma97_umax::Float64 = 1e-4`

#### Mumps (requires `MadNLPMumps`)
- `mumps_dep_tol::Float64 = 0.`
- `mumps_mem_percent::Int = 1000`
- `mumps_permuting_scaling::Int = 7`
- `mumps_pivot_order::Int = 7`
- `mumps_pivtol::Float64 = 1e-6`
- `mumps_pivtolmax::Float64 = .1`
- `mumps_scaling::Int = 77`

#### Umfpack (requires `MadNLPUmfpack`)
- `umfpack_pivtol::Float64 = 1e-4`
- `umfpack_pivtolmax::Float64 = 1e-1`
- `umfpack_sym_pivtol::Float64 = 1e-3`
- `umfpack_block_size::Float64 = 16`
- `umfpack_strategy::Float64 = 2.`

#### Pardiso (requires `MadNLPPardiso`)
- `pardiso_matching_strategy::Pardiso.MatchingStrategy = COMPLETE2x2`
- `pardiso_max_inner_refinement_steps::Int = 1`
- `pardiso_msglvl::Int = 0`
- `pardiso_order::Int = 2`

#### PardisoMKL
- `pardisomkl_num_threads::Int = 1`
- `pardiso_matching_strategy::PardisoMKL.MatchingStrategy = COMPLETE2x2`
- `pardisomkl_max_iterative_refinement_steps::Int = 1`
- `pardisomkl_msglvl::Int = 0`
- `pardisomkl_order::Int = 2`

#### LapackGPU (requires `MadNLPGPU`)
- `lapackgpu_algorithm::LapackGPU.Algorithms = BUNCHKAUFMAN`

#### LapackCPU
- `lapackcpu_algorithm::LapackCPU.Algorithms = BUNCHKAUFMAN`

#### Schur (requires `MadNLPGraphs`)
- `schur_subproblem_solver::Module = DefaultLinearSolver` \
   Linear solver used for solving subproblem. Valid values are: {`MadNLPUmfpack`, `MadNLPMa27`, `MadNLPMa57`, `MadNLPMa97`, `MadNLPMumps`}.
- `schur_dense_solver::Module = DefaultDenseSolver` \
   Linear solver used for solving Schur complement system
- `schur_custom_partition::Bool = false` \
   If `false`, Schur solver automatically detects the partition using `Metis`. If `true`, the partition information given in `schur_part` is used. `schur_num_parts` and `schur_part` should be properly set by the user. When using with `Plasmo`, `schur_num_parts` and `schur_part` are automatically set by the `Plasmo` interface.
- `schur_num_parts::Int = 2` \
   Number of parts (excluding the parent node). Valid range is ``[1,\infty)``
- `schur_part::Vector{Int} = Int[]` \
   Custom partition information in a vector form. The parent node should be labeled as `0`. Only valid if `schur_custom_partition` is `true`.

#### Schwarz (requires `MadNLPGraphs`)
- `schwarz_subproblem_solver::Module = DefaultSubproblemSolver` \
   Linear solver used for solving subproblem. Valid values are: {`MadNLPUmfpack`, `MadNLPPardisoMKL`, `MadNLPMa27`, `MadNLPMa57`, `MadNLPMa77`, `MadNLPMa86`, `MadNLPMa97`, `MadNLPPardiso`}.
- `schwarz_custom_partition::Bool = false` \
    If `false`, Schwarz solver automatically detects the partition using `Metis`. If `true`, the partition information given in `schur_part` is used. `schur_num_parts` and `schur_part` should be properly set by the user. When using with `Plasmo`, `schur_num_parts` and `schur_part` are automatically set by the `Plasmo` interface.
- `schwarz_num_parts::Int = 2` \
    Number of parts. Valid range is ``[1,\infty)``
- `schwarz_part::Vector{Int} = Int[]` \
    Custom partition information in a vector form. Only valid if `schwar_custom_partition` is `true`.
- `schwarz_num_parts_upper::Int = 0` \
    Number of parts in upper level partition. If `schwarz_num_parts_upper!=0`, a bilevel partitioning scheme is used. Valid range is ``[1,\infty)``
- `schwarz_part_upper::Vector{Int} = Int[]` \
    Custom partition for the upper level partition.
- `schwarz_fully_improve_subproblem_solver::Bool = true` \
    If `true`, the subproblem solvers are fully improved when the linear solver is initialized.
- `schwarz_max_expand_factor::Int = 4` \
    The size of overlap is fully saturated when the `improve!` is called `schwarz_max_expand_factor-1` times. Valid range is ``[2,\infty)``.

### Iterator Options
#### Richardson
- `richardson_max_iter::Int = 10` \
    Maximum number of Richardson iteration steps. Valid range is ``[1,\infty)``.
- `richardson_tol::Float64 = 1e-10` \
    Convergence tolerance of Richardson iteration. Valid range is ``(0,\infty)``.
- `richardson_acceptable_tol::Float64 = 1e-5` \
    Acceptable convergence tolerance of Richardson iteration. If the Richardson iteration counter exceeds `richardson_max_iter` without satisfying the convergence criteria set with `richardson_tol`, the Richardson solver checks whether the acceptable convergence criteria set with `richardson_acceptable_tol` is satisfied; if the acceptable convergence criteria is satisfied, the computed step is used; otherwise, the augmented system is treated to be singular. Valid range is ``(0,\infty)``.

#### Krylov (requires `MadNLPIterative`)
- `krylov_max_iter::Int = 10` \
    Maximum number of Krylov iteration steps. Valid range is ``[1,\infty)``.
- `krylov_tol::Float64 = 1e-10` \
    Convergence tolerance of Krylov iteration. Valid range is ``(0,\infty)``.
- `krylov_acceptable_tol::Float64 = 1e-5` \
    Acceptable convergence tolerance of Krylov iteration. If the Krylov iteration counter exceeds `krylov_max_iter` without satisfying the convergence criteria set with `krylov_tol`, the Krylov solver checks whether the acceptable convergence criteria set with `krylov_acceptable_tol` is satisfied; if the acceptable convergence criteria is satisfied, the computed step is used; otherwise, the augmented system is treated to be singular. Valid range is ``(0,\infty)``.
- `krylov_restart::Int = 5` \
    Maximum Krylov iteration before restarting. Valid range is ``[1,\infty)``.

### Reference
[Bunch, 1977]: J R Bunch and L Kaufman, Some stable methods for calculating inertia and solving symmetric linear systems, Mathematics of Computation 31:137 (1977), 163-179.

[Greif, 2014]: Greif, Chen, Erin Moulding, and Dominique Orban. "Bounds on eigenvalues of matrices arising from interior-point methods." SIAM Journal on Optimization 24.1 (2014): 49-83.

[Wächter, 2006]: Wächter, Andreas, and Lorenz T. Biegler. "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming." Mathematical programming 106.1 (2006): 25-57.

[Chiang, 2016]: Chiang, Nai-Yuan, and Victor M. Zavala. "An inertia-free filter line-search algorithm for large-scale nonlinear programming." Computational Optimization and Applications 64.2 (2016): 327-354.
