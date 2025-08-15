testset = [
    # Temporarily commented out since LapackGPUSolver does not currently support sparse callbacks
    [
        "CUDSS",
        ()->MadNLP.Optimizer(
            linear_solver=CUDSSSolver,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "CUDSS-AMD",
        ()->MadNLP.Optimizer(
            linear_solver=CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_ordering=MadNLPGPU.AMD_ORDERING,
        ),
        [],
    ],
    [
        "CUDSS-SYMAMD",
        ()->MadNLP.Optimizer(
            linear_solver=CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_ordering=MadNLPGPU.SYMAMD_ORDERING,
        ),
        [],
    ],
    [
        "CUDSS-COLAMD",
        ()->MadNLP.Optimizer(
            linear_solver=CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_ordering=MadNLPGPU.COLAMD_ORDERING,
        ),
        [],
    ],
    [
        "CUDSS-METIS",
        ()->MadNLP.Optimizer(
            linear_solver=CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_ordering=MadNLPGPU.METIS_ORDERING,
        ),
        [],
    ],
    [
        "CUDSS-IR",
        ()->MadNLP.Optimizer(
            linear_solver=CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_ir=1,
        ),
        [],
    ],
    [
        "CUDSS-HYBRID",
        ()->MadNLP.Optimizer(
            linear_solver=CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_hybrid_memory=true,
        ),
        [],
    ],
    [
        "CUDSS-NOPIVOTING",
        ()->MadNLP.Optimizer(
            linear_solver=CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_pivoting=false,
        ),
        [],
    ],
    # [
    #     "Formulation K2.5",
    #     ()->MadNLP.Optimizer(
    #         linear_solver=CUDSSSolver,
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
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-LU (legacy)",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR,
            legacy=true,
        ),
        [],
    ],
    [
        "LapackGPU-LU",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR,
            legacy=false,
        ),
        [],
    ],
    [
        "LapackGPU-QR (legacy)",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR,
            legacy=true,
        ),
        [],
    ],
    [
        "LapackGPU-QR",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR,
            legacy=false,
        ),
        [],
    ],
    [
        "LapackGPU-CHOLESKY (legacy)",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR,
            legacy=true,
        ),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"], # KKT system not PD
    ],
    [
        "LapackGPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR,
            legacy=false,
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
