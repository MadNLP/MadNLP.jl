import Pkg

MADNLP_GIT_HASH = ARGS[1]
Pkg.add(name="MadNLPHSL", rev="$MADNLP_GIT_HASH")
Pkg.add(name="MadNLPGPU", rev="$MADNLP_GIT_HASH")

@everywhere begin
    using MadNLPBenchmark, Distributed, MadNLPHSL, MadNLPGPU, CUDA
    CUDA.device!(mod(myid(), CUDA.ndevices()))
end

# CUTEST CPU benchmark
run_benchmark(
    "$(MADNLP_GIT_HASH)_cutest_cpu",
    case -> CUTEstModel(case; decode = false),
    CUTEST_CASES,
    m -> madnlp(m; linear_solver=Ma57Solver, ma57_automatic_scaling=true, tol=1e-6, max_wall_time=900.0),
    (case, result) -> (; case=case, stats=result.counter);
    finalizer = m -> finalize(m)
)

# OPF CPU benchmark
run_benchmark(
    "$(MADNLP_GIT_HASH)_opf_cpu",
    case -> opf_model(case)[1],
    OPF_CASES,
    m -> madnlp(m; linear_solver=Ma27Solver, max_wall_time=900.0),
    (case, result) -> (; case=case, stats=result.counter)
)

# COPS CPU benchmark
run_benchmark(
    "$(MADNLP_GIT_HASH)_cops_cpu",
    ((model,args),) -> model(args..., COPSBenchmark.ExaModelsBackend()),
    COPS_CASES,
    m -> madnlp(m; linear_solver=Ma27Solver, max_wall_time=900.0),
    (case, result) -> (; case=case, stats=result.counter)
)

# CUTEST GPU benchmark
run_benchmark(
    "$(MADNLP_GIT_HASH)_cutest_gpu",
    case -> WrapperNLPModel(typeof(CUDA.zeros(Float64,0)), CUTEstModel(case; decode = false)),
    CUTEST_CASES,
    m -> madnlp(m; tol = 1e-8, max_wall_time=900.0),
    (case, result) -> (; case=case, stats=result.counter);
    finalizer = m -> finalize(m.inner)
)

# OPF GPU benchmark
run_benchmark(
    "$(MADNLP_GIT_HASH)_opf_gpu",
    case -> opf_model(case; backend = CUDABackend())[1],
    OPF_CASES,
    m -> madnlp(m; tol = 1e-8, max_wall_time=900.0),
    (case, result) -> (; case=case, stats=result.counter);
    wks = workers()[1:CUDA.ndevices()]
)

# COPS GPU benchmark
run_benchmark(
    "$(MADNLP_GIT_HASH)_cops_gpu",
    ((model,args),) -> model(args..., COPSBenchmark.ExaModelsBackend(); backend = CUDABackend()),
    COPS_CASES,
    m -> madnlp(m; tol = 1e-8, max_wall_time=900.0),
    (case, result) -> (; case=case, stats=result.counter)
)
