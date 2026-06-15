# HSL integration tests, incorporated from the old lib/MadNLPHSL/test. The HSL
# solvers come from MadCoreHSL (reexported by MadNLP). Guarded on a licensed
# HSL_jll being functional: the public CI runners use the stub HSL_jll so this
# skips there; it runs on the HSL self-hosted runner (and locally with a dev'd
# licensed HSL_jll). The standalone linear-solver test lives in MadCoreHSL/test.

import HSL: LIBHSL_isfunctional
import Quadmath: Float128

hsl_testset = [
    ("HSL-Ma27", () -> MadNLP.Optimizer(linear_solver = Ma27Solver, print_level = MadNLP.ERROR), String[]),
    ("HSL-Ma57", () -> MadNLP.Optimizer(linear_solver = Ma57Solver, print_level = MadNLP.ERROR), String[]),
    ("HSL-Ma77", () -> MadNLP.Optimizer(linear_solver = Ma77Solver, print_level = MadNLP.ERROR), ["unbounded"]),
    ("HSL-Ma86", () -> MadNLP.Optimizer(linear_solver = Ma86Solver, print_level = MadNLP.ERROR), String[]),
    ("HSL-Ma97", () -> MadNLP.Optimizer(linear_solver = Ma97Solver, print_level = MadNLP.ERROR), String[]),
]

if LIBHSL_isfunctional()
    for hsl_solver in [Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver]
        @testset "$(nameof(hsl_solver))" begin
            MadNLPTests.test_linear_solver(hsl_solver, Float64)
            if MadNLP.is_supported(hsl_solver, Float128)
                MadNLPTests.test_linear_solver(hsl_solver, Float128)
            end
        end
    end
    for (name, optimizer_constructor, exclude) in hsl_testset
        test_madnlp(name, optimizer_constructor, exclude)
    end
else
    @info "HSL not functional (stub HSL_jll) — skipping HSL integration tests"
    @test_skip true
end
