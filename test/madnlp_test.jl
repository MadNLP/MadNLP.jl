testset = [
    [
        "Umfpack",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.UmfpackSolver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-BUNCHKAUFMAN",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-LU",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-QR",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"]
    ],
    [
        "Option: RELAX_BOUND",
        ()->MadNLP.Optimizer(
            fixed_variable_treatment=MadNLP.RelaxBound,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "Option: AUGMENTED KKT SYSTEM",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.SparseUnreducedKKTSystem,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "Option: SPARSE CONDENSED KKT SYSTEM",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            equality_treatment = MadNLP.RelaxEquality,
            fixed_variable_treatment = MadNLP.RelaxBound,
            tol = 1e-4,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "Option: InertiaFree",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.InertiaFree,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "Option: InertiaFree & AUGMENTED KKT SYSTEM",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.InertiaFree,
            kkt_system=MadNLP.SparseUnreducedKKTSystem,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "Option: InertiaFree & SPARSE CONDENSED KKT SYSTEM",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.InertiaFree,
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            equality_treatment = MadNLP.RelaxEquality,
            fixed_variable_treatment = MadNLP.RelaxBound,
            tol = 1e-4,
            print_level=MadNLP.ERROR),
        []
    ],
]


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
