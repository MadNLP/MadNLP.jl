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
]

@testset "MadNLPGPU test" begin
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude)
    end
end


