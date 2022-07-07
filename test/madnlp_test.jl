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
            lapackcpu_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-LU",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.LapackCPUSolver,
            lapackcpu_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-QR",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.LapackCPUSolver,
            lapackcpu_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.LapackCPUSolver,
            lapackcpu_algorithm=MadNLP.CHOLESKY,
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
    nlp = MadNLPTests.HS15Model()
    ips = MadNLP.InteriorPointSolver(nlp; print_level=MadNLP.ERROR)
    MadNLP.optimize!(ips)
    @test ips.status == MadNLP.SOLVE_SUCCEEDED
end


# @testset "NLS problem" begin
#     nlp = MadNLPTests.NLSModel()
#     ips = MadNLP.InteriorPointSolver(nlp; print_level=MadNLP.ERROR)
#     MadNLP.optimize!(ips)
#     @test ips.status == MadNLP.SOLVE_SUCCEEDED
# end

