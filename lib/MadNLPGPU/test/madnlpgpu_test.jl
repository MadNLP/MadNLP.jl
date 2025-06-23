testset = [
    # Temporarily commented out since LapackGPUSolver does not currently support sparse callbacks
    [
        "CUDSS",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "CUDSS-AMD",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            ordering=MadNLPGPU.AMD_ORDERING,
        ),
        [],
    ],
    [
        "CUDSS-METIS",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            ordering=MadNLPGPU.METIS_ORDERING,
        ),
        [],
    ],
    [
        "CUDSS-HYBRID",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            hybrid=true,
            ir=1,
        ),
        [],
    ],
    [
        "CUDSS-NOPIVOTING",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            pivoting=false,
        ),
        [],
    ],
    # [
    #     "Formulation K2.5",
    #     ()->MadNLP.Optimizer(
    #         linear_solver=MadNLPGPU.CUDSSSolver,
    #         print_level=MadNLP.ERROR,
    #         kkt_system=MadNLP.ScaledSparseKKTSystem,
    #     ),
    #     [],
    # ],
    [
        "LapackGPU-BUNCHKAUFMAN",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "LapackGPU-LU",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "LapackGPU-QR",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "LapackGPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR
        ),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"], # KKT system not PD
    ],
]

@testset "MadNLPGPU test" begin
    MadNLPTests.test_linear_solver(LapackGPUSolver,Float32)
    MadNLPTests.test_linear_solver(LapackGPUSolver,Float64)
    # Test LapackGPU wrapper
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude; Arr=CuArray)
    end
end
