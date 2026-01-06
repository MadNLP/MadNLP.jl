using Pkg, Distributed, JLD2, Logging

MADNLP_GIT_HASH = ARGS[1]

@everywhere begin
    using ExaModelsPower, COPSBenchmark, CUTEst, ExaModels
    using MadNLPHSL, MadNLPGPU, CUDA
    CUDA.device!(mod(myid(), CUDA.ndevices()))
    
    function standard_analyzer(case, result)
        if result == nothing
            return(; case=string(case), status=MadNLP.INTERNAL_ERROR, total_time=Inf, total_iter=0)
        else
            return (; case=string(case), status=Int(result.status), total_time=result.counters.total_time, total_iter=result.counters.k)
        end
    end
end

function run_benchmark(name, model, cases, solver, analyzer; wks=workers(), finalizer=m -> nothing)
    
    results = pmap(WorkerPool(wks), cases) do case
        @info "Running $(name) $case"
        m = model(case)

        result = nothing
        try 
            first_run_time = @timed begin
                result = solver(m)
            end
            
            if first_run_time.compile_time + first_run_time.recompile_time > 0
                @info "$case: running again to avoid compilation time effects"
                result = solver(m)
            end
        catch e
            @info "$case: solver failed with error $e"
        end

        @info "$case: completed"
        finalizer(m)

        return analyzer(case, result)
    end
    JLD2.@save "$name.jld2" name results

    return results
end

@everywhere begin
    using ExaModelsPower, COPSBenchmark, CUTEst, ExaModels
    using MadNLPHSL, MadNLPGPU, CUDA
    CUDA.device!(mod(myid(), CUDA.ndevices()))
end

const OPF_CASES = [
    "pglib_opf_case197_snem.m", "pglib_opf_case300_ieee.m", "pglib_opf_case588_sdet.m",
    "pglib_opf_case2000_goc.m", "pglib_opf_case3012wp_k.m", "pglib_opf_case5_pjm.m",
    "pglib_opf_case200_activ.m", "pglib_opf_case3022_goc.m", "pglib_opf_case60_c.m",
    "pglib_opf_case20758_epigrids.m", "pglib_opf_case30_as.m", "pglib_opf_case6468_rte.m",
    "pglib_opf_case2312_goc.m", "pglib_opf_case30_ieee.m", "pglib_opf_case6470_rte.m",
    "pglib_opf_case2383wp_k.m", "pglib_opf_case3120sp_k.m", "pglib_opf_case6495_rte.m",
    "pglib_opf_case10000_goc.m", "pglib_opf_case240_pserc.m", "pglib_opf_case3375wp_k.m", "pglib_opf_case6515_rte.m",
    "pglib_opf_case10192_epigrids.m", "pglib_opf_case24464_goc.m", "pglib_opf_case3970_goc.m", "pglib_opf_case7336_epigrids.m",
    "pglib_opf_case10480_goc.m", "pglib_opf_case24_ieee_rts.m", "pglib_opf_case39_epri.m", "pglib_opf_case73_ieee_rts.m",
    "pglib_opf_case118_ieee.m", "pglib_opf_case2736sp_k.m", "pglib_opf_case3_lmbd.m", "pglib_opf_case78484_epigrids.m",
    "pglib_opf_case1354_pegase.m", "pglib_opf_case2737sop_k.m", "pglib_opf_case4020_goc.m", "pglib_opf_case793_goc.m",
    "pglib_opf_case13659_pegase.m", "pglib_opf_case2742_goc.m", "pglib_opf_case4601_goc.m", "pglib_opf_case8387_pegase.m",
    "pglib_opf_case14_ieee.m", "pglib_opf_case2746wop_k.m", "pglib_opf_case4619_goc.m", "pglib_opf_case89_pegase.m",
    "pglib_opf_case162_ieee_dtc.m", "pglib_opf_case2746wp_k.m", "pglib_opf_case4661_sdet.m", "pglib_opf_case9241_pegase.m",
    "pglib_opf_case179_goc.m", "pglib_opf_case2848_rte.m", "pglib_opf_case4837_goc.m", "pglib_opf_case9591_goc.m",
    "pglib_opf_case1803_snem.m", "pglib_opf_case2853_sdet.m", "pglib_opf_case4917_goc.m", 
    "pglib_opf_case1888_rte.m", "pglib_opf_case2868_rte.m", "pglib_opf_case500_goc.m", 
    "pglib_opf_case19402_goc.m", "pglib_opf_case2869_pegase.m", "pglib_opf_case5658_epigrids.m",
    "pglib_opf_case1951_rte.m", "pglib_opf_case30000_goc.m", "pglib_opf_case57_ieee.m"
]

const COPS_CASES = [
    (COPSBenchmark.bearing_model, (50, 50)),
    (COPSBenchmark.chain_model, (800,)),
    (COPSBenchmark.camshape_model, (1000,)),
    (COPSBenchmark.catmix_model, (100,)),
    (COPSBenchmark.elec_model, (50,)),
    (COPSBenchmark.gasoil_model, (100,)),
    (COPSBenchmark.marine_model, (100,)),
    (COPSBenchmark.methanol_model, (100,)),
    (COPSBenchmark.minsurf_model, (50, 50)),
    (COPSBenchmark.minsurf_model, (50, 75)),
    (COPSBenchmark.minsurf_model, (50, 100)),
    (COPSBenchmark.pinene_model, (100,)),
    (COPSBenchmark.robot_model, (200,)),
    (COPSBenchmark.rocket_model, (400,)),
    (COPSBenchmark.steering_model, (200,)),
    (COPSBenchmark.dirichlet_model, (20,)),
    (COPSBenchmark.henon_model, (10,)),
    (COPSBenchmark.lane_emden_model, (20,)),
]

const CUTEST_CASES = setdiff(
    CUTEst.select_sif_problems(),
    [
        # MadNLP running into error
        # Ipopt running into error
        "EG3", # lfact blows up
        # Problems that are hopelessly large
        "TAX213322",
        "TAXR213322",
        "TAX53322",
        "TAXR53322",
        "YATP1LS",
        "YATP2LS",
        "YATP1CLS",
        "YATP2CLS",
        "CYCLOOCT",
        "CYCLOOCF",
        "LIPPERT1",
        "GAUSSELM",
        "BA-L52LS",
        "BA-L73LS",
        "BA-L21LS",
        "BA-L52",
        "NONSCOMPNE",
        "LOBSTERZ",
        # Failure
        "CHARDIS0"
    ]
)

try
    @info "Testing CUTEst model loading..."
    m = CUTEstModel(CUTEst.select_sif_problems()[end]; decode = false)
catch e
    @info "CUTEst models could not be loaded. Decoding all CUTEst problems..."
    for (i, instance) in enumerate(CUTEst.select_sif_problems())
        try
            m=CUTEstModel(instance; decode = false)
            @debug "Model $i-th $(instance) loaded successfully."
            finalize(m)
        catch e
            CUTEst.sifdecoder(instance)
            CUTEst.build_libsif(instance)
            m=CUTEstModel(instance; decode = false)
            finalize(m)
            @info "Model $i-th $(instance) decoded and loaded successfully."
        end
    end
end

# CUTEST CPU benchmark
run_benchmark(
    "$(MADNLP_GIT_HASH)_cutest_cpu",
    case -> CUTEstModel(case; decode = false),
    CUTEST_CASES,
    m -> madnlp(m; linear_solver=Ma57Solver, ma57_automatic_scaling=true, tol=1e-6, max_wall_time=900.0, rethrow_error =false),
    standard_analyzer;
    finalizer = m -> finalize(m)
)

# # OPF CPU benchmark
# run_benchmark(
#     "$(MADNLP_GIT_HASH)_opf_cpu",
#     case -> opf_model(case)[1],
#     OPF_CASES,
#     m -> madnlp(m; linear_solver=Ma27Solver, max_wall_time=900.0, rethrow_error =false),
#     standard_analyzer
# )

# # COPS CPU benchmark
# run_benchmark(
#     "$(MADNLP_GIT_HASH)_cops_cpu",
#     ((model,args),) -> model(args..., COPSBenchmark.ExaModelsBackend()),
#     COPS_CASES,
#     m -> madnlp(m; linear_solver=Ma57Solver, ma57_automatic_scaling=true, max_wall_time=900.0, rethrow_error =false),
#     standard_analyzer
# )

# # CUTEST GPU benchmark
# run_benchmark(
#     "$(MADNLP_GIT_HASH)_cutest_gpu",
#     case -> WrapperNLPModel(typeof(CUDA.zeros(Float64,0)), CUTEstModel(case; decode = false)),
#     CUTEST_CASES,
#     m -> madnlp(m; tol = 1e-8, max_wall_time=900.0, rethrow_error =false),
#     standard_analyzer;
#     wks = workers()[1:CUDA.ndevices()],
#     finalizer = m -> finalize(m.inner)
# )

# OPF GPU benchmark
run_benchmark(
    "$(MADNLP_GIT_HASH)_opf_gpu",
    case -> opf_model(case; backend = CUDABackend())[1],
    OPF_CASES,
    m -> madnlp(m; tol = 1e-8, max_wall_time=900.0, rethrow_error =false),
    standard_analyzer;
    wks = workers()[1:CUDA.ndevices()]
)

# COPS GPU benchmark
run_benchmark(
    "$(MADNLP_GIT_HASH)_cops_gpu",
    ((model,args),) -> model(args..., COPSBenchmark.ExaModelsBackend(); backend = CUDABackend()),
    COPS_CASES,
    m -> madnlp(m; tol = 1e-8, max_wall_time=900.0, rethrow_error =false),
    standard_analyzer;
    wks = workers()[1:CUDA.ndevices()]
)
