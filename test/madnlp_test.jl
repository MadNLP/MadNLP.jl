testset = [
    [
        "Umfpack",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPUmfpack,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-BUNCHKAUFMAN",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackCPU,
            lapackcpu_algorithm=MadNLPLapackCPU.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-LU",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackCPU,
            lapackcpu_algorithm=MadNLPLapackCPU.LU,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "LapackCPU-QR",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackCPU,
            lapackcpu_algorithm=MadNLPLapackCPU.QR,
            print_level=MadNLP.ERROR),
        []
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

@isdefined(MadNLPPardisoMKL) && push!(
    testset,
    [
        "PardisoMKL",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPPardisoMKL,
            print_level=MadNLP.ERROR),
        ["eigmina"]
    ]
)

for (name,optimizer_constructor,exclude) in testset
    test_madnlp(name,optimizer_constructor,exclude)
end
