include("nlp_test_include.jl")

sets = [
    [
        ()->MadNLP.Optimizer(
            linear_solver="Umfpack",
            print_level=MadNLP.ERROR),
        [],
        "Umfpack" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="Mumps",
            print_level=MadNLP.ERROR),
        [],
        "Mumps" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="Ma27",
            print_level=MadNLP.ERROR),
        [],
        "Ma27" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="Ma57",
            print_level=MadNLP.ERROR),
        [],
        "Ma57" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="Ma77",
            print_level=MadNLP.ERROR),
        ["unbounded"],
        "Ma77" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="Ma86",
            print_level=MadNLP.ERROR),
        [],
        "Ma86" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="Ma97",
            print_level=MadNLP.ERROR),
        [],
        "Ma97" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="Pardiso",
            print_level=MadNLP.ERROR),
        [],
        "Pardiso" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="PardisoMKL",
            print_level=MadNLP.ERROR),
        [],
        "PardisoMKL" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="LapackMKL",
            lapackmkl_algorithm=MadNLP.LapackMKL.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        [],
        "LapackMKL" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="LapackMKL",
            lapackmkl_algorithm=MadNLP.LapackMKL.LU,
            print_level=MadNLP.ERROR),
        [],
        "LapackMKL" in MadNLP.available_linear_solvers()
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="LapackCUDA",
            lapackcuda_algorithm=MadNLP.LapackCUDA.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        [],
        false
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver="LapackCUDA",
            lapackcuda_algorithm=MadNLP.LapackCUDA.LU,
            print_level=MadNLP.ERROR),
        [],
        false
    ],
    [
        ()->MadNLP.Optimizer(
            fixed_variable_treatment=MadNLP.RELAX_BOUND,
            print_level=MadNLP.ERROR),
        [],
        true
    ],
    [
        ()->MadNLP.Optimizer(
            tol=1e-8,
            reduced_system=false,
            print_level=MadNLP.ERROR),
        ["infeasible"], # numerical error at the end],
        true
    ],
    [
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.INERTIA_FREE,
            print_level=MadNLP.ERROR),
        [],
        true
    ],
    [
        ()->MadNLP.Optimizer(
            iterator=MadNLP.Krylov,
            print_level=MadNLP.ERROR),
        ["unbounded"],
        true
    ],
    [
        ()->MadNLP.Optimizer(
            linear_system_scaler=isdefined(MadNLP,:Mc19) ? MadNLP.Mc19 : MadNLP.DummyModule,
            print_level=MadNLP.ERROR),
        ["eigmina"],
        true
    ],
    [
        ()->MadNLP.Optimizer(
            disable_garbage_collector=true,
            output_file=".test.out"
        ),
        ["infeasible","unbounded","eigmina"], # just checking logger; no need to test all
        "Umfpack" in MadNLP.available_linear_solvers()
    ],
]

@testset "NLP test" for (optimizer_constructor,exclude,availability) in sets
    availability && nlp_test(optimizer_constructor,exclude)
end
