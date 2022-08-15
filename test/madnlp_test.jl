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
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-LU",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-QR",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR),
        ["infeasible", "lootsma", "eigmina"]
    ],
    [
        "Option: RELAX_BOUND",
        ()->MadNLP.Optimizer(
            fixed_variable_treatment=MadNLP.RELAX_BOUND,
            print_level=MadNLP.ERROR),
        [],
        true
    ],
    [
        "Option: AUGMENTED KKT SYSTEM",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.SPARSE_UNREDUCED_KKT_SYSTEM,
            print_level=MadNLP.ERROR),
        ["infeasible","eigmina"] # numerical errors
    ],
    [
        "Option: INERTIA_FREE & AUGMENTED KKT SYSTEM",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.INERTIA_FREE,
            kkt_system=MadNLP.SPARSE_UNREDUCED_KKT_SYSTEM,
            print_level=MadNLP.ERROR),
        ["infeasible","eigmina"] # numerical errors
    ],
    [
        "Option: INERTIA_FREE",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.INERTIA_FREE,
            print_level=MadNLP.ERROR),
        []
    ],
]


for (name,optimizer_constructor,exclude) in testset
    test_madnlp(name,optimizer_constructor,exclude)
end

@testset "HS15 problem" begin
    options = Dict{Symbol, Any}(
        :print_level=>MadNLP.ERROR,
    )
    nlp = MadNLPTests.HS15Model()
    ips = MadNLP.InteriorPointSolver(nlp; option_dict=options)
    MadNLP.optimize!(ips)
    @test ips.status == MadNLP.SOLVE_SUCCEEDED
end


@testset "NLS problem" begin
    options = Dict{Symbol, Any}(
        :print_level=>MadNLP.ERROR,
    )
    nlp = MadNLPTests.NLSModel()
    ips = MadNLP.InteriorPointSolver(nlp; option_dict=options)
    MadNLP.optimize!(ips)
    @test ips.status == MadNLP.SOLVE_SUCCEEDED
end

@testset "MadNLP timings" begin
    nlp = MadNLPTests.HS15Model()
    ips = MadNLP.InteriorPointSolver(nlp)
    time_callbacks = MadNLP.timing_callbacks(ips)
    @test isa(time_callbacks, Array)
    time_linear_solver = MadNLP.timing_linear_solver(ips)
    @test isa(time_linear_solver, Array)
    time_madnlp = MadNLP.timing_madnlp(ips)
    @test isa(time_madnlp.time_linear_solver, Array)
    @test isa(time_madnlp.time_callbacks, Array)
end

