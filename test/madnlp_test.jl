testset = [
    [
        "SparseKKTSystem + Mumps",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.MumpsSolver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseKKTSystem + Umfpack",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.UmfpackSolver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseKKTSystem + InertiaFree",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.InertiaFree,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseKKTSystem + RelaxBound",
        ()->MadNLP.Optimizer(
            fixed_variable_treatment=MadNLP.RelaxBound,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "ScaledSparseKKTSystem + LapackCPU",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.LapackCPUSolver,
            kkt_system=MadNLP.ScaledSparseKKTSystem,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "DenseKKTSystem + LapackCPU-BUNCHKAUFMAN",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "DenseKKTSystem + LapackCPU-LU",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "DenseKKTSystem + LapackCPU-QR",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "DenseKKTSystem + LapackCPU-EVD",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.EVD,
            print_level=MadNLP.ERROR),
        [
            "eigmina" # fails; regularization does not correct the inertia; inertia calculation based on EVD does not seem reliable
         ]
    ],
    [
        "DenseKKTSystem + LapackCPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"]
    ],
    [
        "SparseUnreducedKKTSystem",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.SparseUnreducedKKTSystem,
            linear_solver=UmfpackSolver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseUnreducedKKTSystem + InertiaFree",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.InertiaFree,
            kkt_system=MadNLP.SparseUnreducedKKTSystem,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseCondensedKKTSystem + CHOLMOD-CHOLESKY",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            equality_treatment = MadNLP.RelaxEquality,
            fixed_variable_treatment = MadNLP.RelaxBound,
            linear_solver=MadNLP.CHOLMODSolver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseCondensedKKTSystem + InertiaFree",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.InertiaFree,
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            equality_treatment = MadNLP.RelaxEquality,
            fixed_variable_treatment = MadNLP.RelaxBound,
            print_level=MadNLP.ERROR),
        []
    ],
]

# N.B. Current CHOLMOD interface is supported only starting from Julia v1.10.
if VERSION >= v"1.10"
    push!(
        testset,
        [
            "SparseCondensedKKTSystem + CHOLMOD-LDL",
            ()->MadNLP.Optimizer(
                kkt_system=MadNLP.SparseCondensedKKTSystem,
                equality_treatment = MadNLP.RelaxEquality,
                fixed_variable_treatment = MadNLP.RelaxBound,
                linear_solver=MadNLP.CHOLMODSolver,
                cholmod_algorithm=MadNLP.LDL,
                print_level=MadNLP.ERROR),
            []
        ]
    )
end

for (name,optimizer_constructor,exclude) in testset
    test_madnlp(name,optimizer_constructor,exclude)
end

@testset "HS15 problem" begin
    nlp = MadNLPTests.HS15Model()
    n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)
    x0 = NLPModels.get_x0(nlp)
    y0 = NLPModels.get_y0(nlp)

    # Test all combinations between x0 and y0
    for xini in [nothing, x0], yini in [nothing, y0]
        solver = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.ERROR)
        MadNLP.solve!(solver; x=xini, y=yini)
        @test solver.status == MadNLP.SOLVE_SUCCEEDED
    end

    # Test all arguments at the same time
    zl = zeros(n)
    zu = zeros(n)
    solver = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.ERROR)
    MadNLP.solve!(solver; x=x0, y=y0, zl=zl, zu=zu)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
end

@testset "Fixed variables" begin
    nlp = MadNLPTests.HS15Model()
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR)
    MadNLP.solve!(solver)
    @test isa(solver.cb.fixed_handler, MadNLP.NoFixedVariables)

    # Fix first variable:
    nlp.meta.lvar[1] = 0.5
    solver_sparse = MadNLP.MadNLPSolver(nlp; callback=MadNLP.SparseCallback, print_level=MadNLP.ERROR)
    sol_sparse = MadNLP.solve!(solver_sparse)
    @test length(solver_sparse.ind_fixed) == 1
    @test isa(solver_sparse.cb.fixed_handler, MadNLP.MakeParameter)
    @test solver_sparse.n == 3 # fixed variables are removed

    solver_dense = MadNLP.MadNLPSolver(nlp; callback=MadNLP.DenseCallback, print_level=MadNLP.ERROR)
    sol_dense = MadNLP.solve!(solver_dense)
    @test length(solver_dense.ind_fixed) == 1
    @test isa(solver_dense.cb.fixed_handler, MadNLP.MakeParameter)
    @test solver_dense.n == 4 # fixed variables are frozen

    @test sol_dense.iter == sol_sparse.iter
    @test sol_dense.objective == sol_sparse.objective
    @test sol_dense.solution == sol_sparse.solution
    @test sol_dense.multipliers == sol_sparse.multipliers
    @test sol_dense.multipliers_L == sol_sparse.multipliers_L
    @test sol_dense.multipliers_U == sol_sparse.multipliers_U
end

@testset "MadNLP warmstart" begin
    nlp = MadNLPTests.HS15Model()
    x0 = NLPModels.get_x0(nlp)
    y0 = NLPModels.get_y0(nlp)

    solver = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.ERROR)
    @test solver.status == MadNLP.INITIAL
    MadNLP.solve!(solver; x=x0, y=y0)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
    # Update barrier term and solve again
    MadNLP.solve!(solver; x=x0, y=y0, mu_init=1e-5)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
end


@testset "NLS problem" begin
    nlp = MadNLPTests.NLSModel()
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR)
    MadNLP.solve!(solver)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
end

@testset "MadNLP callback allocations" begin
    nlp = MadNLPTests.HS15Model()
    solver = MadNLPSolver(nlp)
    MadNLP.initialize!(solver)
    kkt = solver.kkt
    x, f, c = solver.x, solver.f, solver.c
    # Precompile
    MadNLP.eval_f_wrapper(solver, x)
    MadNLP.eval_grad_f_wrapper!(solver, f, x)
    MadNLP.eval_cons_wrapper!(solver, c, x)
    MadNLP.eval_jac_wrapper!(solver, kkt, x)
    MadNLP.eval_lag_hess_wrapper!(solver, kkt, x, solver.y)

    n_allocs = @allocated MadNLP.eval_f_wrapper(solver, x)
    @test n_allocs == 16 # objective is still allocating
    n_allocs = @allocated MadNLP.eval_grad_f_wrapper!(solver, f, x)
    @test n_allocs == 0
    n_allocs = @allocated MadNLP.eval_cons_wrapper!(solver, c, x)
    @test n_allocs == 0
    n_allocs = @allocated MadNLP.eval_jac_wrapper!(solver, kkt, x)
    @test n_allocs == 0
    n_allocs = @allocated MadNLP.eval_lag_hess_wrapper!(solver, kkt, x, solver.y)
    @test n_allocs == 0
end

@testset "MadNLP timings" begin
    nlp = MadNLPTests.HS15Model()
    solver = MadNLPSolver(nlp)
    MadNLP.initialize!(solver)
    time_callbacks = MadNLP.timing_callbacks(solver)
    @test isa(time_callbacks, NamedTuple)
    time_linear_solver = MadNLP.timing_linear_solver(solver)
    @test isa(time_linear_solver, NamedTuple)
    time_madnlp = MadNLP.timing_madnlp(solver)
    @test isa(time_madnlp.time_linear_solver, NamedTuple)
    @test isa(time_madnlp.time_callbacks, NamedTuple)
end

@testset "Quadmath test" begin
    nlp = MadNLPTests.HS15Model(T = Float128)
    result = madnlp(
        nlp;
        print_level = MadNLP.ERROR,
        callback = MadNLP.SparseCallback,
        linear_solver=LDLSolver,
        kkt_system = MadNLP.SparseCondensedKKTSystem
    )
    @test result.status == MadNLP.SOLVE_SUCCEEDED
end

@testset "Issue #430" begin
    # Test MadNLP is working with bound_relax_factor=0
    nlp = MadNLPTests.HS15Model()
    solver = MadNLPSolver(nlp; bound_relax_factor=0.0, print_level=MadNLP.ERROR)
    stats = MadNLP.solve!(solver)
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
end

