using MadNLP, MadNLPHSL, MadNLPGPU, SIMDiffExamples, CUDA
using JuMP, Ipopt
using StatsPlots, LaTeXStrings
pgfplotsx()

SIMDiffExamples.silence()
CUDA.allowscalar(false)

tol=1e-5
save = []

for (path,casename) in (
    # (ENV["PGLIB_PATH"], "pglib_opf_case89_pegase.m"),
    # (ENV["PGLIB_PATH"], "pglib_opf_case1354_pegase.m"),
    # (ENV["PGLIB_PATH"], "pglib_opf_case2869_pegase.m"),
    # (ENV["PGLIB_PATH"], "pglib_opf_case8387_pegase.m"),
    (ENV["PGLIB_PATH"], "pglib_opf_case9241_pegase.m"),
    # (ENV["PGLIB_PATH"], "pglib_opf_case13659_pegase.m"),
    # ("/home/sshin/git/PowerSystemsTestData/ACTIVSg2000/", "case_ACTIVSg2000.m"),
    # ("/home/sshin/git/PowerSystemsTestData/ACTIVSg10k/", "case_ACTIVSg10k.m"),
    # ("/home/sshin/git/PowerSystemsTestData/ACTIVSg70k/", "case_ACTIVSg70k.m"),
    )

    case = joinpath(path, casename)

    m = MadNLP.ScaledNLPModel(SIMDiffExamples.ac_power_model(case, CuArray, CUDABackend()))

    m.meta.uvar .+= tol
    m.meta.ucon .+= tol

    t11 = @elapsed begin
        madnlp(m; kkt_system=MadNLP.SparseCondensedKKTSystem, linear_solver=MadNLPGPU.RFSolver, tol=tol)
    end
    t21 = sum(getfield(m.inner.counters, t) for t in [:teval_obj, :teval_obj, :teval_obj, :teval_obj, :teval_obj])

    # m = MadNLP.ScaledNLPModel(SIMDiffExamples.ac_power_model(case))
    # m.meta.uvar .+= tol
    # m.meta.ucon .+= tol
    # t11 = @elapsed begin
    #     s = MadNLPSolver(m; linear_solver=Ma27Solver, tol=tol)
    #     MadNLP.initialize!(s.kkt)
    #     solve!(s)
    # end
    # t21 = sum(getfield(m.inner.counters, t) for t in [:teval_obj, :teval_obj, :teval_obj, :teval_obj, :teval_obj]) 

    m = SIMDiffExamples.jump_ac_power_model(case)
    set_optimizer(m, Ipopt.Optimizer)
    set_optimizer_attribute(m, "linear_solver", "ma27")
    set_optimizer_attribute(m, "tol", tol)
    t12 = @elapsed optimize!(m)
    t22 = sum(getfield(m.moi_backend.optimizer.model.counters, t) for t in [:teval_obj, :teval_cons, :teval_grad, :teval_jac, :teval_hess])
    push!(
        save,
        (casename, t11, t12, t21, t22)
    )
end



stime = vcat([[t11-t21, t13-t23] for (casename, t11, t12, t21, t22) in save]...)
ftime = vcat([[t21, t23] for (casename, t11, t12, t21, t22) in save]...)
ticks = vcat([["SIMDiff + MadNLP (GPU)", "JuMP + Ipopt (CPU)"] for (casename, t11, t12, t21, t22) in save]...)

plt = groupedbar(
    [stime ftime],
    bar_position = :stack,
    bar_width=0.7,
    xticks=(1:12, ticks),
    framestyle=:box,
    label=["Solver Internal + Linear Algebra" "Derivative Evaluation"],
    ylim=(0,1.1*maximum(stime.+ftime)),
    ylabel="Wall Clock Time (sec)",
    xrotation = 60,
    legend=:topleft,
    size=(600,300),
)

savefig(plt, "bar.pdf")
