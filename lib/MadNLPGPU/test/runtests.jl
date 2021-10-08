using Test, MadNLP, MadNLPGPU, MadNLPTests

testset = [
    [
        "LapackGPU-BUNCHKAUFMAN",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackGPU,
            lapackgpu_algorithm=MadNLPLapackGPU.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPLapackGPU)
    ],
    [
        "LapackGPU-LU",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackGPU,
            lapackgpu_algorithm=MadNLPLapackGPU.LU,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPLapackGPU)
    ],
    [
        "LapackGPU-QR",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackGPU,
            lapackgpu_algorithm=MadNLPLapackGPU.QR,
            print_level=MadNLP.ERROR),
        [],
        @isdefined(MadNLPLapackGPU)
    ],
    [
        "LapackGPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPLapackGPU,
            lapackgpu_algorithm=MadNLPLapackGPU.CHOLESKY,
            print_level=MadNLP.ERROR),
        ["infeasible", "lootsma", "eigmina"],
        @isdefined(MadNLPLapackGPU)
    ],
]

# Test LapackGPU wrapper
@testset "LapackGPU test" begin
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude)
    end
end

# Test DenseKKTSystem on GPU
include("densekkt_gpu.jl")

