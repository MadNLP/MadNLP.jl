using MadNLP, MadNLPHSL, MadNLPCUSOLVER, SIMDiffExamples; SIMDiffExamples.silence()

tol = 1e-3

case = 
    # "/home/sshin/git/pglib-opf/pglib_opf_case30_ieee.m"
    # "/home/sshin/git/pglib-opf/pglib_opf_case300_ieee.m"
    "/home/sshin/git/pglib-opf/pglib_opf_case1354_pegase.m"
    # "/home/sshin/git/pglib-opf/pglib_opf_case2869_pegase.m"
    # "/home/sshin/git/pglib-opf/pglib_opf_case9241_pegase.m"
    # "/home/sshin/git/pglib-opf/pglib_opf_case13659_pegase.m"
    # "/home/sshin/git/PowerSystemsTestData/ACTIVSg70k/case_ACTIVSg70k.m"

m = SIMDiffExamples.ac_power_model(case)
s2 = MadNLPSolver(
    m;
    linear_solver=Ma27Solver,
    tol = tol,
    # kkt_system=MadNLP.SPARSE_CONDENSED_KKT_SYSTEM,
    # fixed_variable_treatment = MadNLP.RELAX_BOUND
)
@time MadNLP.solve!(s2)

m.meta.ucon .+= tol
s1 = MadNLPSolver(
    m;
    # linear_solver=Ma27Solver,
    linear_solver=MadNLPCUSOLVER.RFSolver,
    tol = tol,
    kkt_system=MadNLP.SPARSE_CONDENSED_KKT_SYSTEM,
    fixed_variable_treatment = MadNLP.RELAX_BOUND
)
@time MadNLP.solve!(s1)


