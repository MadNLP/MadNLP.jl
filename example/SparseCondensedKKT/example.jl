using MadNLP, MadNLPHSL, MadNLPGPU, SIMDiffExamples, CUDA
CUDA.allowscalar(false)

tol=1e-5
case =
    "/home/sshin/git/pglib-opf/pglib_opf_case9241_pegase.m"
    # "/home/sshin/git/pglib-opf/pglib_opf_case13659_pegase.m"
    # "/home/sshin/git/PowerSystemsTestData/ACTIVSg70k/case_ACTIVSg70k.m"

m = MadNLP.ScaledNLPModel(SIMDiffExamples.ac_power_model(case, CuArray, CUDABackend()))

m.meta.uvar .+= tol
m.meta.ucon .+= tol

@time s1=MadNLPSolver(m; kkt_system=MadNLP.SparseCondensedKKTSystem, linear_solver=MadNLPGPU.RFSolver, tol=tol)
MadNLP.initialize!(s1.kkt)
solve!(s1)

# m = MadNLP.ScaledNLPModel(SIMDiffExamples.ac_power_model(case, Array, CPU()))
# m.meta.uvar .+= tol
# m.meta.ucon .+= tol
# @time s2=MadNLPSolver(m; kkt_system=MadNLP.SparseCondensedKKTSystem, linear_solver=Ma27Solver, tol=tol, max_iter=1, print_level=MadNLP.DEBUG)
# MadNLP.initialize!(s2.kkt)
# solve!(s2)
