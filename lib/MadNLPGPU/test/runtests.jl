using Test, MadNLP, MadNLPGPU, MadNLPTests

testset = [
    [
        "LapackGPU-BUNCHKAUFMAN",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapackgpu_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        [],
    ],
    [
        "LapackGPU-LU",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapackgpu_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR),
        [],
    ],
    [
        "LapackGPU-QR",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapackgpu_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR),
        [],
    ],
    [
        "LapackGPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapackgpu_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR),
        ["infeasible", "lootsma", "eigmina"],
    ],
]

@testset "MadNLPGPU test" begin

    MadNLPTests.test_linear_solver(LapackGPUSolver,Float32)
    MadNLPTests.test_linear_solver(LapackGPUSolver,Float64)

    # Test LapackGPU wrapper
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude)
    end
end

# Test DenseKKTSystem on GPU
include("densekkt_gpu.jl")
