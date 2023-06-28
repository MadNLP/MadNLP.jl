using MadNLP, MadNLPHSL, MadNLPGPU, SIMDiffExamples, CUDA, MadNLPCUSOLVER, NLPModelsIpopt, SparseArrays
CUDA.allowscalar(false)

tol=1e-4
m = SIMDiffExamples.ac_power_model("/home/sshin/git/pglib-opf/pglib_opf_case1354_pegase.m", CuArray, CUDABackend())
m.meta.uvar .+= tol
m.meta.ucon .+= tol

@time s=MadNLPSolver(m; kkt_system=MadNLP.SparseCondensedKKTSystem, linear_solver=MadNLPCUSOLVER.RFSolver2, tol=tol)
solve!(s)
