cuda_testset = [
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
    #     "Formulation K2",
    #     ()->MadNLP.Optimizer(
    #         linear_solver=CUDSSSolver,
    #         print_level=MadNLP.ERROR,
    #         kkt_system=MadNLP.SparseKKTSystem,
    #     ),
    #     [],
    # ],
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
        "LapackGPU-LU",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-QR",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR,
        ),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"], # KKT system not PD
    ],
]

rocm_testset = [
    [
        "LapackROCSolver-LU",
        ()->MadNLP.Optimizer(
            linear_solver=LapackROCSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackROCSolver-QR",
        ()->MadNLP.Optimizer(
            linear_solver=LapackROCSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackROCSolver-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver=LapackROCSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR,
        ),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"], # KKT system not PD
    ],
]

oneapi_testset = [
    [
        "LapackOneMKLSolver-LU",
        ()->MadNLP.Optimizer(
            linear_solver=LapackOneMKLSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    # [
    #     "LapackOneMKLSolver-QR",
    #     ()->MadNLP.Optimizer(
    #         linear_solver=LapackOneMKLSolver,
    #         lapack_algorithm=MadNLP.QR,
    #         print_level=MadNLP.ERROR,
    #     ),
    #     [],
    # ],
    # [
    #     "LapackOneMKLSolver-CHOLESKY",
    #     ()->MadNLP.Optimizer(
    #         linear_solver=LapackOneMKLSolver,
    #         lapack_algorithm=MadNLP.CHOLESKY,
    #         print_level=MadNLP.ERROR,
    #     ),
    #     ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"], # KKT system not PD
    # ],
]
@testset "MadNLPGPU test" begin
    # if CUDA.functional()
    #     MadNLPTests.test_linear_solver(LapackGPUSolver,Float32)
    #     MadNLPTests.test_linear_solver(LapackGPUSolver,Float64)
    #     # Test LapackGPU wrapper
    #     for (name,optimizer_constructor,exclude) in cuda_testset
    #         test_madnlp(name,optimizer_constructor,exclude; Arr=CuArray)
    #     end
    # end
    # if AMDGPU.functional()
    #     MadNLPTests.test_linear_solver(LapackROCSolver,Float32)
    #     MadNLPTests.test_linear_solver(LapackROCSolver,Float64)
    #     for (name,optimizer_constructor,exclude) in rocm_testset
    #         test_madnlp(name,optimizer_constructor,exclude; Arr=ROCArray)
    #     end
    # end
    if oneAPI.functional()
        MadNLPTests.test_linear_solver(LapackOneMKLSolver,Float32)
        MadNLPTests.test_linear_solver(LapackOneMKLSolver,Float64)
        for (name,optimizer_constructor,exclude) in oneapi_testset
            test_madnlp(name,optimizer_constructor,exclude; Arr=oneArray)
        end
    end
end
