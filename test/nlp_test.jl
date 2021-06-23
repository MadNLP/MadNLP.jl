include("nlp_test_include.jl")

@test begin
    local m,x
    m=Model(MadNLP.Optimizer)
    @variable(m,x)
    @objective(m,Min,x^2)
    MOIU.attach_optimizer(m)

    nlp = MadNLP.NonlinearProgram(m.moi_backend.optimizer.model)
    ips = MadNLP.Solver(nlp)
    
    show(stdout, "text/plain",nlp)
    show(stdout, "text/plain",ips)
    true
end


sets = [
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPUmfpack,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPUmfpack)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMumps,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPMumps)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMa27,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPMa27)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMa57,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPMa57)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMa77,
            print_level=MadNLP.ERROR),
        ["unbounded"],
        @isdefined(MadNLPMa77)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMa86,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPMa86)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPMa97,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPMa97)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPPardiso,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPPardiso)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPPardisoMKL,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPPardisoMKL)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackCPU,
            lapackcpu_algorithm=MadNLPLapackCPU.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPLapackCPU)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackCPU,
            lapackcpu_algorithm=MadNLPLapackCPU.LU,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPLapackCPU)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackCPU,
            lapackcpu_algorithm=MadNLPLapackCPU.QR,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPLapackCPU)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackGPU,
            lapackgpu_algorithm=MadNLPLapackGPU.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPLapackGPU)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackGPU,
            lapackgpu_algorithm=MadNLPLapackGPU.LU,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPLapackGPU)
    ],
    [
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackGPU,
            lapackgpu_algorithm=MadNLPLapackGPU.QR,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPLapackGPU)
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
            reduced_system=false,
            print_level=MadNLP.ERROR),
        ["infeasible","eigmina"], # numerical errors
        true
    ],
    [
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.INERTIA_FREE,
            reduced_system=false),
        ["infeasible","eigmina"], # numerical errors
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
            iterator=MadNLPKrylov,
            print_level=MadNLP.ERROR),
        ["unbounded"],
        @isdefined(MadNLPKrylov)
    ],
    [
        ()->MadNLP.Optimizer(
            disable_garbage_collector=true,
            output_file=".test.out"
        ),
        ["infeasible","unbounded","eigmina"], # just checking logger; no need to test all
        true
    ],
]

@testset "NLP test" for (optimizer_constructor,exclude,availability) in sets
    availability && nlp_test(optimizer_constructor,exclude)
end
